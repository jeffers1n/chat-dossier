import json
import ssl
import urllib.error
import unittest
from pathlib import Path
from unittest.mock import patch
from core import parse_chat, validate_analysis, analyze, chunk_messages, tls_context, http_error_message, connection_error_message, deduplicate

class CoreTests(unittest.TestCase):
    def test_telegram_rich_text_and_service(self):
        rows={'messages':[{'id':7,'date':'2026-10-20','from':'A','text':['Привет ',{'type':'bold','text':'мир'}]},{'type':'service','text':'ignore'}]}
        self.assertEqual(parse_chat(json.dumps(rows))[0], {'id':'m1','source_id':'7','date':'2026-10-20','author':'A','forwarded_from':None,'text':'Привет мир'})
    def test_forwarded_author_is_separate(self):
        rows={'messages':[{'id':8,'date':'2026-01-01','from':'Анна','forwarded_from':'Илья','text':'Обещаю закончить к пятнице'}]}
        message=parse_chat(json.dumps(rows,ensure_ascii=False))[0]
        self.assertEqual(message['author'],'Анна')
        self.assertEqual(message['forwarded_from'],'Илья')
        self.assertEqual(message['text'],'Обещаю закончить к пятнице')

    def test_plain(self):
        self.assertEqual(parse_chat('[сегодня] Илья: Сделаю')[0]['author'],'Илья')
        self.assertEqual(parse_chat('просто строка')[0]['text'],'просто строка')
    def test_empty_invalid_and_limits(self):
        for value in ['', '{}', '[bad', None, 'x'*600001, '\n'.join('x' for _ in range(1201))]:
            with self.subTest(value=str(value)[:10]), self.assertRaises(ValueError):parse_chat(value)
    def test_reject_fabricated_evidence_and_single_source_conflict(self):
        messages=parse_chat('Да\nНет')
        events=[{'category':'decision','title':'t','evidence':[{'message_id':'m1','quote':'Да'}]}, {'category':'decision','title':'t','evidence':[{'message_id':'m9','quote':'Да'}]}, {'category':'decision','title':'t','evidence':[{'message_id':'m1','quote':'Выдумка'}]}, {'category':'contradiction','title':'t','evidence':[{'message_id':'m1','quote':'Да'}]}]
        result=validate_analysis({'events':events},messages)
        self.assertEqual((len(result['events']),result['rejected']),(1,3))
    def test_demo_all_quotes_valid(self):
        root=Path(__file__).resolve().parents[1]
        result=validate_analysis(json.loads((root/'fixtures/demo.json').read_text()),parse_chat((root/'fixtures/demo.txt').read_text()))
        self.assertEqual(result['rejected'],0)
        self.assertEqual(len(result['events']),6)
    def test_chunking_preserves_ids(self):
        messages=parse_chat('\n'.join(f'[{i}] A: ' + 'x'*70 for i in range(20)))
        chunks=chunk_messages(messages,500)
        self.assertGreater(len(chunks),1)
        self.assertEqual([m['id'] for c in chunks for m in c],[m['id'] for m in messages])
    @patch.dict('os.environ', {'LLM_API_KEY':''})
    def test_no_key(self):
        with self.assertRaisesRegex(ValueError,'LLM_API_KEY'):analyze(parse_chat('test'))
    def test_multiple_parts_are_combined(self):
        from unittest.mock import MagicMock
        messages=parse_chat('\n'.join(f'[{i}] A: ' + 'x'*70 for i in range(400)))
        calls=[]
        def response(request, timeout, context=None):
            payload=json.loads(request.data)
            chunk=json.loads(payload['messages'][1]['content'])
            calls.append(chunk)
            first=chunk[0]
            item={'category':'decision','title':'t','evidence':[{'message_id':first['id'],'quote':first['text'][:10]}]}
            mock=MagicMock()
            mock.__enter__.return_value.read.return_value=json.dumps({'choices':[{'message':{'content':json.dumps({'events':[item]})}}]}).encode()
            return mock
        with patch.dict('os.environ',{'LLM_API_KEY':'test'}),patch('urllib.request.urlopen',side_effect=response):
            result=analyze(messages)
        self.assertGreater(len(calls),1)
        self.assertEqual(result['parts']+1,len(calls))
        self.assertEqual(len(result['events']),result['parts'])
        self.assertEqual([m['id'] for c in calls[:-1] for m in c],[m['id'] for m in messages])
        self.assertEqual(result['reconciliation'],'complete')

    def test_global_429_keeps_validated_chunk_results(self):
        from unittest.mock import MagicMock
        messages=parse_chat('\n'.join(f'[{i}] A: ' + 'x'*70 for i in range(400)))
        calls=[]
        def response(request, timeout, context=None):
            rows=json.loads(json.loads(request.data)['messages'][1]['content'])
            calls.append(rows)
            if len(rows)==len(messages):
                raise urllib.error.HTTPError('https://example.test',429,'Quota',{},None)
            first=rows[0]
            event={'category':'decision','title':'Решение','evidence':[{'message_id':first['id'],'quote':first['text'][:10]}]}
            mock=MagicMock()
            mock.__enter__.return_value.read.return_value=json.dumps({'choices':[{'message':{'content':json.dumps({'events':[event]})}}]}).encode()
            return mock
        with patch.dict('os.environ',{'LLM_API_KEY':'test'}),patch('urllib.request.urlopen',side_effect=response):
            result=analyze(messages)
        self.assertEqual(len(calls),result['parts']+1)
        self.assertEqual(result['reconciliation'],'skipped_rate_limited')
        self.assertEqual(len(result['events']),result['parts'])
        self.assertEqual(result['rejected'],0)

    def test_chunk_429_still_fails_without_claiming_completeness(self):
        error=urllib.error.HTTPError('https://example.test',429,'Quota',{},None)
        with patch.dict('os.environ',{'LLM_API_KEY':'test'}),patch('urllib.request.urlopen',side_effect=error):
            with self.assertRaisesRegex(ValueError,'HTTP 429'):
                analyze(parse_chat('text'))

    def test_cross_chunk_change_and_closed_question(self):
        from unittest.mock import MagicMock
        rows=['Кто согласует дату?','Решили запуск 10 июня.']
        rows += ['Фоновое обсуждение ' + 'x'*6000 for _ in range(5)]
        rows += ['Согласовали перенос запуска на 12 июня.','Я согласую дату.']
        messages=parse_chat('\n'.join(rows))
        calls=[]
        def response(request, timeout, context=None):
            payload=json.loads(request.data)
            chunk=json.loads(payload['messages'][1]['content'])
            calls.append(chunk)
            if len(chunk)==len(messages):
                events=[{'category':'change','title':'Запуск перенесён','evidence':[{'message_id':'m2','quote':'Решили запуск 10 июня.'},{'message_id':'m8','quote':'Согласовали перенос запуска на 12 июня.'}]}]
            else:
                events=[]
                if chunk[0]['id']=='m1':
                    events=[{'category':'question','title':'Кто согласует дату','evidence':[{'message_id':'m1','quote':'Кто согласует дату?'}]}]
            mock=MagicMock()
            mock.__enter__.return_value.read.return_value=json.dumps({'choices':[{'message':{'content':json.dumps({'events':events})}}]}).encode()
            return mock
        with patch.dict('os.environ',{'LLM_API_KEY':'test'}),patch('urllib.request.urlopen',side_effect=response):
            result=analyze(messages)
        self.assertGreater(result['parts'],1)
        self.assertEqual(result['reconciliation'],'complete')
        self.assertEqual([(e['category'],[x['message_id'] for x in e['evidence']]) for e in result['events']],[('change',['m2','m8'])])
        self.assertEqual(result['rejected'],0)

    def test_deduplication_uses_category_and_source_ids(self):
        one={'category':'change','title':'Первый вариант','evidence':[{'message_id':'m1','quote':'a'},{'message_id':'m2','quote':'b'}]}
        duplicate={'category':'change','title':'Другой заголовок','evidence':[{'message_id':'m2','quote':'b'},{'message_id':'m1','quote':'a'}]}
        distinct={'category':'contradiction','title':'Другое событие','evidence':one['evidence']}
        self.assertEqual(deduplicate([one,duplicate,distinct]),[one,distinct])

    def test_reconciliation_limit_is_reported(self):
        from unittest.mock import MagicMock
        import core
        messages=parse_chat('\n'.join('x'*5000 for _ in range(7)))
        count=[]
        def response(request, timeout, context=None):
            count.append(1)
            mock=MagicMock()
            mock.__enter__.return_value.read.return_value=json.dumps({'choices':[{'message':{'content':json.dumps({'events':[]})}}]}).encode()
            return mock
        with patch.dict('os.environ',{'LLM_API_KEY':'test'}),patch.object(core,'RECONCILE_CHARS',100),patch('urllib.request.urlopen',side_effect=response):
            result=analyze(messages)
        self.assertGreater(result['parts'],1)
        self.assertEqual(result['reconciliation'],'skipped_too_large')
        self.assertEqual(len(count),result['parts'])

    def test_incomplete_model_response_fails_closed(self):
        from unittest.mock import MagicMock
        mock=MagicMock()
        mock.__enter__.return_value.read.return_value=json.dumps({'choices':[{'finish_reason':'length','message':{'content':'{"events":[]}'}}]}).encode()
        with patch.dict('os.environ',{'LLM_API_KEY':'test'}),patch('urllib.request.urlopen',return_value=mock):
            with self.assertRaisesRegex(ValueError,'неполный ответ'):
                analyze(parse_chat('Привет'))

    def test_temporary_503_retries_and_succeeds(self):
        from unittest.mock import MagicMock
        failure=urllib.error.HTTPError('https://example.test',503,'Unavailable',{},None)
        success=MagicMock()
        success.__enter__.return_value.read.return_value=b'{"choices":[{"message":{"content":"{\\\"events\\\":[]}"}}]}'
        with patch.dict('os.environ',{'LLM_API_KEY':'test'}), patch('urllib.request.urlopen',side_effect=[failure,success]) as call, patch('time.sleep') as pause:
            self.assertEqual(analyze(parse_chat('text'))['events'],[])
        self.assertEqual(call.call_count,2)
        pause.assert_called_once_with(1)

    def test_repeated_503_reports_attempts_without_partial_result(self):
        failures=[urllib.error.HTTPError('https://example.test',503,'Unavailable',{},None) for _ in range(4)]
        with patch.dict('os.environ',{'LLM_API_KEY':'test'}), patch('urllib.request.urlopen',side_effect=failures) as call, patch('time.sleep') as pause:
            with self.assertRaisesRegex(ValueError,'После 4 попыток'):
                analyze(parse_chat('text'))
        self.assertEqual(call.call_count,4)
        self.assertEqual([x.args[0] for x in pause.call_args_list],[1,2,4])

    def test_gemini_compatibility_request(self):
        from unittest.mock import MagicMock
        observed={}
        def response(request, timeout, context=None):
            observed['url']=request.full_url
            observed['payload']=json.loads(request.data)
            mock=MagicMock()
            mock.__enter__.return_value.read.return_value=b'{"choices":[{"message":{"content":"{\\"events\\":[]}"}}]}'
            return mock
        with patch.dict('os.environ',{'GEMINI_API_KEY':'test','LLM_API_KEY':'','LLM_BASE_URL':'','LLM_MODEL':''}),patch('urllib.request.urlopen',side_effect=response):
            # Empty optional overrides are ignored by the compatibility defaults.
            analyze(parse_chat('text'))
        self.assertEqual(observed['url'],'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions')
        self.assertEqual(observed['payload']['model'],'gemini-3.6-flash')
        self.assertEqual(observed['payload']['response_format']['type'],'json_schema')

    def test_custom_model_override(self):
        from unittest.mock import MagicMock
        observed = {}
        def response(request, timeout, context=None):
            observed.update(json.loads(request.data))
            mock = MagicMock()
            mock.__enter__.return_value.read.return_value = json.dumps({'choices':[{'message':{'content':json.dumps({'events':[]})}}]}).encode()
            return mock
        with patch.dict('os.environ', {'GEMINI_API_KEY':'test','LLM_API_KEY':'','LLM_MODEL':'custom-model'}), patch('urllib.request.urlopen', side_effect=response):
            analyze(parse_chat('text'))
        self.assertEqual(observed['model'], 'custom-model')

    def test_connection_diagnostics_and_verified_tls(self):
        self.assertEqual(tls_context().verify_mode, ssl.CERT_REQUIRED)
        self.assertIn('квота', http_error_message(429))
        self.assertIn('Ключ', http_error_message(401))
        self.assertIn('доступа', http_error_message(403))
        error = urllib.error.URLError(ssl.SSLCertVerificationError('certificate verify failed'))
        self.assertIn('сертификат', connection_error_message(error))

    def test_api_response(self):
        from unittest.mock import MagicMock
        response=MagicMock()
        response.__enter__.return_value.read.return_value=json.dumps({'choices':[{'message':{'content':json.dumps({'events':[]})}}]}).encode()
        with patch.dict('os.environ',{'LLM_API_KEY':'test'}),patch('urllib.request.urlopen',return_value=response):
            result = analyze(parse_chat('text'))
            self.assertEqual(result['events'],[])
            self.assertEqual(result['model'],'gpt-4o-mini')

if __name__=='__main__':unittest.main()
