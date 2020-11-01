version 1.0

import "Structs.wdl"

workflow GATKSVGenotype {
  input {
    File vcf
    File sample_coverage_file
    Int records_per_shard = 2000
    String batch

    String genotyping_gatk_docker
    String sharding_gatk_docker
    String sv_base_mini_docker

    # cpu or cuda
    String device_train = "cpu"
    String device_genotype = "cpu"

    String gpu_type = "nvidia-tesla-k80"
    String nvidia_driver_version = "450.80.02"

    RuntimeAttr? runtime_attr_split
    RuntimeAttr? runtime_attr_shard
    RuntimeAttr? runtime_attr_train
    RuntimeAttr? runtime_attr_infer
    RuntimeAttr? runtime_attr_concat
  }

  call SplitDepthCalls {
    input:
      vcf = vcf,
      vcf_index = vcf + ".tbi",
      output_name = batch,
      gatk_docker = sharding_gatk_docker,
      runtime_attr_override = runtime_attr_split
  }

  # Note INV records were converted to BND by SVCluster
  Array[String] svtypes = ["DEL", "DUP", "INS", "BND"]

  scatter (svtype in svtypes) {
    call ShardVcf {
      input:
        vcf = SplitDepthCalls.out_pesr,
        vcf_index = SplitDepthCalls.out_pesr_index,
        records_per_shard = records_per_shard,
        svtype = svtype,
        basename = batch + "." + svtype,
        gatk_docker = sharding_gatk_docker,
        runtime_attr_override = runtime_attr_shard
    }
    scatter (i in range(length(ShardVcf.out))) {
      String model_name = batch + ".shard_" + i
      call SVTrainGenotyping {
        input:
          vcf = ShardVcf.out[i],
          sample_coverage_file = sample_coverage_file,
          model_name = model_name,
          gatk_docker = genotyping_gatk_docker,
          device = device_train,
          gpu_type = gpu_type,
          nvidia_driver_version = nvidia_driver_version
      }
      call SVGenotype {
        input:
          vcf = ShardVcf.out[i],
          model_tar = SVTrainGenotyping.out,
          model_name = model_name,
          output_vcf_filename = "~{model_name}.genotyped.vcf.gz",
          gatk_docker = genotyping_gatk_docker,
          device = device_genotype,
          gpu_type = gpu_type,
          nvidia_driver_version = nvidia_driver_version
      }
    }
  }

  Array[File] genotyped_vcf_shards = flatten([[SplitDepthCalls.out_depth], SVGenotype.out])
  Array[File] genotyped_vcf_shard_indexes = flatten([[SplitDepthCalls.out_depth_index], SVGenotype.out_index])

  call ConcatVcfs {
    input:
      vcfs = genotyped_vcf_shards,
      vcfs_idx = genotyped_vcf_shard_indexes,
      merge_sort = true,
      outfile_prefix = "~{batch}.final",
      sv_base_mini_docker = sv_base_mini_docker,
      runtime_attr_override = runtime_attr_concat
  }

  output {
    File genotyped_vcf = ConcatVcfs.out
    File genotyped_vcf_index = ConcatVcfs.out_index
  }
}

task SVTrainGenotyping {
  input {
    File vcf
    File sample_coverage_file
    String model_name
    String gatk_docker
    String device
    Int? max_iter
    String gpu_type
    String nvidia_driver_version
    RuntimeAttr? runtime_attr_override
  }

  RuntimeAttr default_attr = object {
    cpu_cores: 1,
    mem_gb: 15,
    disk_gb: 100,
    boot_disk_gb: 20,
    preemptible_tries: 0,
    max_retries: 0
  }
  RuntimeAttr runtime_attr = select_first([runtime_attr_override, default_attr])

  Float mem_gb = select_first([runtime_attr.mem_gb, default_attr.mem_gb])
  Int java_mem_mb = ceil(mem_gb * 1000 * 0.8)

  output {
    File out = "~{model_name}.svgenotype_model.tar.gz"
    Array[File] journals = glob("gatkStreamingProcessJournal-*.txt")
  }
  command <<<
    set -euo pipefail
    mkdir svmodel
    tabix ~{vcf}

    gatk --java-options -Xmx~{java_mem_mb}M SVTrainGenotyping \
      --variant ~{vcf} \
      --coverage-file ~{sample_coverage_file} \
      --output-name ~{model_name} \
      --output-dir svmodel \
      --device ~{device} \
      --jit \
      ~{"--max-iter " + max_iter} \
      --enable-journal

    tar czf ~{model_name}.sv_genotype_model.tar.gz svmodel/*
  >>>
  runtime {
    cpu: select_first([runtime_attr.cpu_cores, default_attr.cpu_cores])
    memory: mem_gb + " GiB"
    disks: "local-disk " + select_first([runtime_attr.disk_gb, default_attr.disk_gb]) + " HDD"
    bootDiskSizeGb: select_first([runtime_attr.boot_disk_gb, default_attr.boot_disk_gb])
    docker: gatk_docker
    preemptible: select_first([runtime_attr.preemptible_tries, default_attr.preemptible_tries])
    maxRetries: select_first([runtime_attr.max_retries, default_attr.max_retries])
    #gpuCount: 1
    #gpuType: gpu_type
    #nvidiaDriverVersion: nvidia_driver_version
    #zones: "us-east1-d us-east1-c us-central1-a us-central1-c us-west1-b"
  }
}

task SVGenotype {
  input {
    File vcf
    File model_tar
    Int predictive_samples = 1000
    Int discrete_samples = 1000
    String model_name
    String output_vcf_filename
    String gatk_docker
    String device
    String gpu_type
    String nvidia_driver_version
    RuntimeAttr? runtime_attr_override
  }

  RuntimeAttr default_attr = object {
    cpu_cores: 1,
    mem_gb: 15,
    disk_gb: 100,
    boot_disk_gb: 20,
    preemptible_tries: 0,
    max_retries: 0
  }
  RuntimeAttr runtime_attr = select_first([runtime_attr_override, default_attr])

  Float mem_gb = select_first([runtime_attr.mem_gb, default_attr.mem_gb])
  Int java_mem_mb = ceil(mem_gb * 1000 * 0.8)

  output {
    File out = "~{output_vcf_filename}"
    File out_index = "~{output_vcf_filename}.tbi"
    Array[File] journals = glob("gatkStreamingProcessJournal-*.txt")
  }
  command <<<

    set -eo pipefail
    source activate gatk
    mkdir svmodel
    tar xzf ~{model_tar} svmodel/
    tabix ~{vcf}

    gatk --java-options -Xmx~{java_mem_mb}M SVGenotype \
      -V ~{vcf} \
      --output ~{output_vcf_filename} \
      --predictive-samples ~{predictive_samples} \
      --discrete-samples ~{discrete_samples} \
      --model-name ~{model_name} \
      --model-dir svmodel \
      --device ~{device} \
      --jit \
      --enable-journal
  >>>
  runtime {
    cpu: select_first([runtime_attr.cpu_cores, default_attr.cpu_cores])
    memory: mem_gb + " GiB"
    disks: "local-disk " + select_first([runtime_attr.disk_gb, default_attr.disk_gb]) + " HDD"
    bootDiskSizeGb: select_first([runtime_attr.boot_disk_gb, default_attr.boot_disk_gb])
    docker: gatk_docker
    preemptible: select_first([runtime_attr.preemptible_tries, default_attr.preemptible_tries])
    maxRetries: select_first([runtime_attr.max_retries, default_attr.max_retries])
    #gpuCount: 1
    #gpuType: gpu_type
    #nvidiaDriverVersion: nvidia_driver_version
    #zones: "us-east1-d us-east1-c us-central1-a us-central1-c us-west1-b"
  }
}

task SplitDepthCalls {
  input {
    File vcf
    File vcf_index
    String output_name
    String gatk_docker
    RuntimeAttr? runtime_attr_override
  }

  RuntimeAttr default_attr = object {
    cpu_cores: 1,
    mem_gb: 3.75,
    disk_gb: 10,
    boot_disk_gb: 10,
    preemptible_tries: 3,
    max_retries: 1
  }
  RuntimeAttr runtime_attr = select_first([runtime_attr_override, default_attr])

  Float mem_gb = select_first([runtime_attr.mem_gb, default_attr.mem_gb])
  Int java_mem_mb = ceil(mem_gb * 1000 * 0.8)

  output {
    File out_depth = "~{output_name}.depth.vcf.gz"
    File out_depth_index = "~{output_name}.depth.vcf.gz.tbi"
    File out_pesr = "~{output_name}.pesr.vcf.gz"
    File out_pesr_index = "~{output_name}.pesr.vcf.gz.tbi"
  }
  command <<<
    set -euo pipefail

    gatk --java-options -Xmx~{java_mem_mb}M SelectVariants \
      -V ~{vcf} \
      -O ~{output_name}.depth.vcf.gz \
      -select "ALGORITHMS == 'depth'"

    gatk --java-options -Xmx~{java_mem_mb}M SelectVariants \
      -V ~{vcf} \
      -O ~{output_name}.pesr.vcf.gz \
      -select "ALGORITHMS == 'depth'" \
      --invertSelect
  >>>
  runtime {
    cpu: select_first([runtime_attr.cpu_cores, default_attr.cpu_cores])
    memory: mem_gb + " GiB"
    disks: "local-disk " + select_first([runtime_attr.disk_gb, default_attr.disk_gb]) + " HDD"
    bootDiskSizeGb: select_first([runtime_attr.boot_disk_gb, default_attr.boot_disk_gb])
    docker: gatk_docker
    preemptible: select_first([runtime_attr.preemptible_tries, default_attr.preemptible_tries])
    maxRetries: select_first([runtime_attr.max_retries, default_attr.max_retries])
  }
}

# Combine multiple sorted VCFs
task ConcatVcfs {
  input {
    Array[File] vcfs
    Array[File]? vcfs_idx
    Boolean merge_sort = false
    String? outfile_prefix
    String sv_base_mini_docker
    RuntimeAttr? runtime_attr_override
  }

  String outfile_name = outfile_prefix + ".vcf.gz"
  String merge_flag = if merge_sort then "--allow-overlaps" else ""

  # when filtering/sorting/etc, memory usage will likely go up (much of the data will have to
  # be held in memory or disk while working, potentially in a form that takes up more space)
  Float input_size = size(vcfs, "GB")
  Float compression_factor = 5.0
  Float base_disk_gb = 5.0
  Float base_mem_gb = 2.0
  RuntimeAttr runtime_default = object {
    mem_gb: base_mem_gb + compression_factor * input_size,
    disk_gb: ceil(base_disk_gb + input_size * (2.0 + compression_factor)),
    cpu_cores: 1,
    preemptible_tries: 3,
    max_retries: 1,
    boot_disk_gb: 10
  }
  RuntimeAttr runtime_override = select_first([runtime_attr_override, runtime_default])
  runtime {
    memory: "~{select_first([runtime_override.mem_gb, runtime_default.mem_gb])} GB"
    disks: "local-disk ~{select_first([runtime_override.disk_gb, runtime_default.disk_gb])} HDD"
    cpu: select_first([runtime_override.cpu_cores, runtime_default.cpu_cores])
    preemptible: select_first([runtime_override.preemptible_tries, runtime_default.preemptible_tries])
    maxRetries: select_first([runtime_override.max_retries, runtime_default.max_retries])
    docker: sv_base_mini_docker
    bootDiskSizeGb: select_first([runtime_override.boot_disk_gb, runtime_default.boot_disk_gb])
  }

  command <<<
    set -euo pipefail
    VCFS="~{write_lines(vcfs)}"
    if ~{!defined(vcfs_idx)}; then
      cat ${VCFS} | xargs -n1 tabix
    fi
    bcftools concat -a ~{merge_flag} --output-type z --file-list ${VCFS} --output "~{outfile_name}"
    tabix -p vcf -f "~{outfile_name}"
  >>>

  output {
    File out = outfile_name
    File out_index = outfile_name + ".tbi"
  }
}


task ShardVcf {
  input {
    File vcf
    File vcf_index
    String svtype
    Int records_per_shard
    String basename
    String gatk_docker
    RuntimeAttr? runtime_attr_override
  }

  RuntimeAttr default_attr = object {
    cpu_cores: 1,
    mem_gb: 3.75,
    disk_gb: 10,
    boot_disk_gb: 10,
    preemptible_tries: 3,
    max_retries: 1
  }
  RuntimeAttr runtime_attr = select_first([runtime_attr_override, default_attr])

  Float mem_gb = select_first([runtime_attr.mem_gb, default_attr.mem_gb])
  Int java_mem_mb = ceil(mem_gb * 1000 * 0.8)

  output {
    Array[File] out = glob("*.vcf.gz")
  }
  command <<<

    set -euo pipefail
    gatk --java-options -Xmx~{java_mem_mb}M SelectVariants \
      -V ~{vcf} \
      -O ~{basename} \
      -select "SVTYPE == '~{svtype}'" \
      --max-variants-per-shard ~{records_per_shard}

  >>>
  runtime {
    cpu: select_first([runtime_attr.cpu_cores, default_attr.cpu_cores])
    memory: select_first([runtime_attr.mem_gb, default_attr.mem_gb]) + " GiB"
    disks: "local-disk " + select_first([runtime_attr.disk_gb, default_attr.disk_gb]) + " HDD"
    bootDiskSizeGb: select_first([runtime_attr.boot_disk_gb, default_attr.boot_disk_gb])
    docker: gatk_docker
    preemptible: select_first([runtime_attr.preemptible_tries, default_attr.preemptible_tries])
    maxRetries: select_first([runtime_attr.max_retries, default_attr.max_retries])
  }
}