"""Run: python3 app.py. Local-only HTTP server."""
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from core import parse_chat, validate_analysis, analyze, GEMINI_MODEL

ROOT = Path(__file__).resolve().parent
MAX_REQUEST_BYTES = 4_000_000


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Do not log chat content or API credentials.

    def respond(self, status, body, content_type='application/json; charset=utf-8'):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        routes = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'), '/style.css': ('style.css', 'text/css')}
        if self.path == '/api/status':
            return self.respond(200, {'configured': bool(os.environ.get('LLM_API_KEY') or os.environ.get('GEMINI_API_KEY')), 'model': os.environ.get('LLM_MODEL', GEMINI_MODEL if os.environ.get('GEMINI_API_KEY') else 'gpt-4o-mini')})
        if self.path not in routes:
            return self.respond(404, {'error': 'Не найдено'})
        name, mime = routes[self.path]
        self.respond(200, (ROOT / 'static' / name).read_bytes(), mime + '; charset=utf-8')

    def do_POST(self):
        # JSON + same-origin checks prevent another site from triggering paid requests.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.respond(403, {'error': 'Запрос разрешен только из этого приложения.'})
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.respond(415, {'error': 'Ожидается application/json.'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size < 0 or size > MAX_REQUEST_BYTES:
                raise ValueError('Запрос слишком большой.')
            data = json.loads(self.rfile.read(size))
            if self.path == '/api/demo':
                content = (ROOT / 'fixtures/demo.txt').read_text()
                messages = parse_chat(content)
                result = validate_analysis(json.loads((ROOT / 'fixtures/demo.json').read_text()), messages)
                mode = 'demo'
            elif self.path == '/api/analyze':
                if not isinstance(data, dict):
                    raise ValueError('Ожидается JSON объект.')
                messages = parse_chat(data.get('content'))
                result = analyze(messages)
                mode = 'live'
            else:
                return self.respond(404, {'error': 'Не найдено'})
            self.respond(200, {**result, 'messages': messages, 'mode': mode})
        except (ValueError, TypeError) as exc:
            self.respond(400, {'error': str(exc)})
        except Exception:
            self.respond(500, {'error': 'Внутренняя ошибка сервера. Проверьте настройки и повторите запрос.'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    print(f'Детектив чатов: http://127.0.0.1:{port}')
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
