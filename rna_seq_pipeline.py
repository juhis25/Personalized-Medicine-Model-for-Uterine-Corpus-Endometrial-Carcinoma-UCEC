import os
import subprocess
import argparse


def run_command(command):
    print(f"\n[Executing]: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")


def run_fastqc(read1, read2, outdir):
    print("[Step 1] Running FastQC...")
    run_command(f"fastqc {read1} -o {outdir}")
    run_command(f"fastqc {read2} -o {outdir}")


def run_star(index_dir, read1, read2, output_prefix):
    print("[Step 2] Aligning with STAR...")
    run_command(
        f"STAR --genomeDir {index_dir} "
        f"--readFilesIn {read1} {read2} "
        f"--runThreadN 4 "
        f"--outFileNamePrefix {output_prefix} "
        f"--outSAMtype BAM SortedByCoordinate"
    )


def run_featurecounts(gtf_file, bam_file, output_file):
    print("[Step 3] Quantifying with featureCounts...")
    run_command(
        f"featureCounts -a {gtf_file} -o {output_file} {bam_file}"
    )


def main():
    parser = argparse.ArgumentParser(description="Basic RNA-Seq pipeline")
    parser.add_argument('--read1', required=True, help="Read 1 FASTQ file")
    parser.add_argument('--read2', required=True, help="Read 2 FASTQ file")
    parser.add_argument('--index', required=True, help="STAR genome index path")
    parser.add_argument('--gtf', required=True, help="GTF annotation file")
    parser.add_argument('--outdir', required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    run_fastqc(args.read1, args.read2, args.outdir)

    prefix = os.path.join(args.outdir, "star_")
    run_star(args.index, args.read1, args.read2, prefix)

    bam_path = prefix + "Aligned.sortedByCoord.out.bam"
    counts_output = os.path.join(args.outdir, "gene_counts.txt")
    run_featurecounts(args.gtf, bam_path, counts_output)

    print("\n✅ Pipeline completed successfully.")


if __name__ == "__main__":
    main()
    