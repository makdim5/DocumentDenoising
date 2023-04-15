import shutil
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer

import logging


class ImageDenoiserHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length'))
        post_body = self.rfile.read(content_len)

        from denoising import denoise_image, TEMP_IMG_PATH

        logging.info("Getting data ... ")
        with open(TEMP_IMG_PATH, "wb") as file:
            file.write(post_body)

        logging.info("Data denoising ... ")
        denoise_image(TEMP_IMG_PATH)

        logging.info("Data sending ... ")
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.end_headers()
        with open("new.png", 'rb') as content:
            shutil.copyfileobj(content, self.wfile)


def run(server_class=HTTPServer, handler_class=BaseHTTPRequestHandler):
    server_address = ('0.0.0.0', 80)
    httpd = server_class(server_address, handler_class)
    try:
        logging.info(f"Server is running at {server_address}")
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%d-%b-%y %H:%M:%S')
    file_handler = logging.FileHandler(filename='app.log', mode='w')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(stream_handler)

    run(handler_class=ImageDenoiserHandler)
