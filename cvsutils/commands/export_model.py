import argparse
import pathlib
import uuid
import requests
import tenacity
from ..common import Environment
from ..training_api import TrainingApi

EXPORT_TYPES = {
    'coreml': ('coreml', None),
    'coreml_fp16': ('coreml', 'coremlfloat16'),
    'ivs': ('ivs', None),
    'tensorflow': ('tensorflow', None),
    'tensorflow_savedmodel': ('tensorflow', 'tensorflowsavedmodel'),
    'tensorflow_lite': ('tensorflow', 'tensorflowlite'),
    'tensorflow_lite_fp16': ('tensorflow', 'tensorflowlitefloat16'),
    'tensorflow_js': ('tensorflow', 'tensorflowjs'),
    'onnx': ('onnx', None),
    'onnx_fp16': ('onnx', 'onnxfloat16'),
    'openvino': ('openvino', None),
    'openvino_no_postprocess': ('openvino', 'NoPostProcess'),
    'vaidk': ('vaidk', None)
}


@tenacity.retry(retry=tenacity.retry_if_exception_type(tenacity.TryAgain), wait=tenacity.wait_fixed(3))
def get_exported_url(training_api, project_id, iteration_id, platform, flavor):
    response = training_api.get_exports(project_id, iteration_id, platform, flavor)
    if not response or response['status'] == 'Failed':
        raise RuntimeError(f"Failed to export. response={response}")
    elif response['status'] == 'Done':
        return response['url']
    elif response['status'] == 'Exporting':
        print('.', end='', flush=True)
        raise tenacity.TryAgain
    else:
        raise RuntimeError(f"Unexpected response: {response}")


def export_model(env, project_id, iteration_id, export_type, output_filename, force):
    training_api = TrainingApi(env)
    platform, flavor = EXPORT_TYPES[export_type]

    response = training_api.get_exports(project_id, iteration_id, platform, flavor)
    if not response or force:
        training_api.export_iteration(project_id, iteration_id, platform, flavor)
        print("Export requested")

    url = get_exported_url(training_api, project_id, iteration_id, platform, flavor)

    print(f"Downloading from {url}")
    response = requests.get(url)
    response.raise_for_status()
    model = response.content

    output_filename.write_bytes(model)

    print(f"Saved to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Export a model from Custom Vision Service")
    parser.add_argument('project_id', type=uuid.UUID, help="Project Id")
    parser.add_argument('iteration_id', type=uuid.UUID, help="Iteration Id")
    parser.add_argument('export_type', help="Export type", choices=EXPORT_TYPES.keys())
    parser.add_argument('--output', type=pathlib.Path, help="Output file path")
    parser.add_argument('--force', action='store_true', help="Requests new export even if the model is already exported.")

    args = parser.parse_args()
    if not args.output:
        args.output = pathlib.Path(f"{args.iteration_id}_{args.export_type}.zip")

    if args.output.exists():
        parser.error(f"{args.output} already exists")

    export_model(Environment(), args.project_id, args.iteration_id, args.export_type.lower(), args.output, args.force)


if __name__ == '__main__':
    main()
