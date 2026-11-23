import urllib.request
import webbrowser

URL = 'http://127.0.0.1:7860'

def main():
    try:
        with urllib.request.urlopen(URL + '/config', timeout=2) as response:
            import json
            running = json.load(response)
    except OSError:
        from app import run
        run()
        return
    if running.get('title') != 'Ảnh thành 3D':
        raise RuntimeError('Cổng 7860 đang được ứng dụng khác sử dụng.')
    webbrowser.open(URL)

if __name__ == '__main__':
    main()
