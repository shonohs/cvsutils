import argparse
import os
import requests
import tempfile
import time
import uuid
import zipfile
from ..common import Environment
from ..training_api import TrainingApi

EXPORT_TYPES = {
    'tensorflow': ('tensorflow', None),
    'onnx': ('onnx', None),
    'coreml': ('coreml', None)
}

def get_ext(binary, platform):
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, 'model.bin')
        with open(filename, 'wb') as f:
            f.write(binary)
        if zipfile.is_zipfile(filename):
            return 'zip'

    if platform == 'tensorflow':
        return 'pb'
    elif platform == 'onnx':
        return 'onnx'
    elif platform == 'coreml':
        return 'mlmodel'

    raise NotImplementedError


def export_model(env, project_id, iteration_id, export_type, output_filename):
    if output_filename and os.path.exists(output_filename):
        raise RuntimeError(f"{output_filename} already exists")
    if export_type not in EXPORT_TYPES:
        raise NotImplementedError(f"Unsupported model type: {export_type}")

    platform, flavor = EXPORT_TYPES[export_type]

    training_api = TrainingApi(env)

    requested = False
    while True:
        response = training_api.get_exports(project_id, iteration_id, platform, flavor)
        if not response:
            if not requested:
                training_api.export_iteration(project_id, iteration_id, platform, flavor)['status']
                print("Export requested")
                requested = True
            else:
                raise RuntimeError
        elif response['status'] == 'Done':
            url = response['url']
            break
        elif response['status'] == 'Failed':
            raise RuntimeError("Failed to export")
        elif response['status'] == 'Exporting':
            print(".", end='', flush=True)
        else:
            raise NotImplementedError(f"Unsupported status: {response['status']}")

        time.sleep(3)

    assert url
    response = requests.get(url)
    response.raise_for_status()
    model = response.content

    if not output_filename:
        ext = get_ext(model, platform)
        output_filename = f"{iteration_id}_{platform}_{flavor}.{ext}"

    with open(output_filename, 'wb') as f:
        f.write(model)

    print(f"Saved to {output_filename}")


def main():
    parser = argparse.ArgumentParser("Export a model from Custom Vision Service")
    parser.add_argument('project_id', type=str, help="Project Id")
    parser.add_argument('iteration_id', type=str, help="Iteration Id")
    parser.add_argument('type', type=str, help="Export type", choices=EXPORT_TYPES.keys())
    parser.add_argument('--output', type=str, help="Output file path")

    args = parser.parse_args()
    export_model(Environment(), uuid.UUID(args.project_id), uuid.UUID(args.iteration_id), args.type.lower(), args.output)

if __name__ == '__main__':
    main()
