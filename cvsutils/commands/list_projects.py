import argparse
from ..common import Environment
from ..training_api import TrainingApi


def list_projects(env, verbose):
    training_api = TrainingApi(env)

    projects = training_api.get_projects()
    for project in projects:
        print(f"{project['id']}: {project['name']}. Created: {project['created_at']} Modified: {project['modified_at']}")
        if verbose:
            iterations = training_api.get_iterations(project['id'])
            if iterations:
                print("Iterations:")
                for iteration in iterations:
                    print(f"    {iteration['id']}: {iteration['name']}. Domain: {iteration['domain_id']}. Published: {iteration['publish_name']}")


def main():
    parser = argparse.ArgumentParser(description="Show a list of projects")
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()
    list_projects(Environment(), args.verbose)


if __name__ == '__main__':
    main()
