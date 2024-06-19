import argparse

from datasets import load_dataset, concatenate_datasets, load_from_disk


def main(args):
    dss = []
    for dataset_path in args.dataset:
        # dataset = load_dataset(dataset_path, split="train", data_files="*.arrow")
        dataset = load_from_disk(dataset_path)
        dss.append(dataset)

    ds = concatenate_datasets(dss)
    ds = ds.shuffle()
    ds.save_to_disk(args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, action="append")
    parser.add_argument("-o", "--out_dir", type=str)
    args = parser.parse_args()
    main(args)

# python merge_datasets.py -d /gpfs/gibbs/pi/gerstein/xt86/bioagent/data/Mol-Instructions/data/Molecule-oriented_Instructions/forward_reaction_prediction/train -d /gpfs/gibbs/pi/gerstein/xt86/bioagent/data/Mol-Instructions/data/Molecule-oriented_Instructions/reagent_prediction/train -d /gpfs/gibbs/pi/gerstein/xt86/bioagent/data/Mol-Instructions/data/Molecule-oriented_Instructions/retrosynthesis/train -d /gpfs/gibbs/pi/gerstein/xt86/bioagent/data/Mol-Instructions/data/Molecule-oriented_Instructions/pretrain_multi_molecule -o /gpfs/gibbs/pi/gerstein/xt86/bioagent/data/Mol-Instructions/data/Molecule-oriented_Instructions/merged