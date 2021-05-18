from argparse import ArgumentParser
import base64
import datetime
import hashlib
import logging
from io import BytesIO
from pathlib import Path

from flask import Flask
from flask import request
from flask import jsonify
from PIL import Image

from predict import predict


CAPTAIN_EMAIL = 'liaoweiskewer0703@gmail.com'
SALT = 'pj323guan302'
app = Flask(__name__)
Path("logs/").mkdir(parents=True, exist_ok=True)


def generate_server_uuid(input_string):
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def base64_to_pil_img(image_64_encoded) -> Image:
    img_base64_binary = image_64_encoded.encode("utf-8")
    img_binary = base64.b64decode(img_base64_binary)
    image = Image.open(BytesIO(img_binary))
    return image


@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    infer_start_ts = datetime.datetime.now().timestamp()
    logger = logging.getLogger("inference")

    data = request.get_json(force=True)
    # esun_timestamp = data['esun_timestamp']

    image_64_encoded = data['image']
    image = base64_to_pil_img(image_64_encoded)

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)
    server_timestamp = int(t.timestamp())

    image_path = f"logs/{t.strftime('%m%d_%H%M%S')}_{data['esun_uuid'][:4]}.jpg"
    image.save(image_path)
    logger.info(f"image saved: {image_path}")

    answer = None
    try:
        answer = predict(image)
        logger.info(f"prediction: {answer}")
    except Exception as e:
        logger.error(e)
    finally:
        if not answer or not isinstance(answer, str):
            answer = "isnull"

    logger.info(f"approx. time spent: {datetime.datetime.now().timestamp() - infer_start_ts} s")

    return jsonify({
        'esun_uuid': data['esun_uuid'],
        'server_uuid': server_uuid,
        'server_timestamp': server_timestamp,
        'answer': answer,
    })


if __name__ == "__main__":
    logging.basicConfig(
        filename=f"logs/server.log",
        level=logging.INFO,
        format="%(asctime)-15s %(levelname)-7s %(name)-10s %(message)s",
    )

    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=8080, help='port')
    arg_parser.add_argument('-d', '--debug', default=True, help='debug')
    options = arg_parser.parse_args()

    app.run(debug=options.debug, port=options.port)
