import argparse
import pathlib
import requests
import time
import uuid
from ..common import Environment
from ..training_api import TrainingApi

EXPORT_TYPES = {
    'tensorflow': ('tensorflow', None),
    'tensorflowlite': ('tensorflow', 'tensorflowlite'),
    'onnx': ('onnx', None),
    'coreml': ('coreml', None),
    'openvino': ('openvino', None)
}


def export_model(env, project_id, iteration_id, export_type, output_filename):
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

    output_filename.write_bytes(model)

    print(f"Saved to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Export a model from Custom Vision Service")
    parser.add_argument('project_id', type=str, help="Project Id")
    parser.add_argument('iteration_id', type=str, help="Iteration Id")
    parser.add_argument('export_type', type=str, help="Export type", choices=EXPORT_TYPES.keys())
    parser.add_argument('--output', type=pathlib.Path, help="Output file path")

    args = parser.parse_args()
    if not args.output:
        args.output = pathlib.Path(f"{args.iteration_id}_{args.export_type}.zip")

    if args.output.exists():
        parser.error(f"{args.output} already exists")

    export_model(Environment(), uuid.UUID(args.project_id), uuid.UUID(args.iteration_id), args.export_type.lower(), args.output)


if __name__ == '__main__':
    main()
