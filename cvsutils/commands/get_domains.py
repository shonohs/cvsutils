from ..common import Environment
from ..training_api import TrainingApi


def get_domains(env):
    training_api = TrainingApi(env)
    domains = training_api.get_domains()

    for domain in domains:
        print(f"id: {domain['id']}, name: {domain['name']}, type: {domain['type']}")


def main():
    get_domains(Environment())


if __name__ == '__main__':
    main()
