import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from app import Handler


class AppTests(unittest.TestCase):
    def test_cyrillic_export_is_not_rejected_by_http_byte_limit(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            export = json.dumps({'messages': [{'text': 'Решили запускать проект.'}], 'metadata': 'я' * 400_000}, ensure_ascii=False)
            body = json.dumps({'content': export}, ensure_ascii=False).encode()
            self.assertGreater(len(body), 650_000)
            request = urllib.request.Request(
                f'http://127.0.0.1:{server.server_port}/api/analyze', body,
                {'Content-Type': 'application/json'})
            fake_result = {'events': [], 'rejected': 0, 'parts': 1, 'reconciliation': 'not_needed'}
            with patch('app.analyze', return_value=fake_result):
                with urllib.request.urlopen(request, timeout=5) as response:
                    result = json.load(response)
            self.assertEqual(result['mode'], 'live')
            self.assertEqual(result['messages'][0]['text'], 'Решили запускать проект.')
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
