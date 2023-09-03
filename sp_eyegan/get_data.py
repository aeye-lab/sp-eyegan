from __future__ import annotations

import argparse

import pymovements as pm


def get_data(dataset_name: str) -> int:
    dataset = pm.Dataset(dataset_name, path=f'data/{dataset_name}')
    dataset.download()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, help='specific which dataset to download')
    args = parser.parse_args()

    get_data(args.dataset_name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
