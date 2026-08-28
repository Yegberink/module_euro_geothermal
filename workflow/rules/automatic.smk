"""Rules to download automatic resource files for geothermal."""

import re


CURL_ARGS = "--fail --silent --show-error --location --retry 5 --retry-delay 5 --retry-all-errors --continue-at -"
AUTOMATIC_DATASETS = internal["resources"]["automatic"]["datasets"]


rule download_stable_dataset:
    output:
        ensure(
            "<resources>/automatic/{dataset}",
            md5=lambda wc: AUTOMATIC_DATASETS[wc.dataset]["md5"],
        ),
    log:
        "<logs>/automatic/download_stable_dataset_{dataset}.log",
    wildcard_constraints:
        dataset="|".join(re.escape(name) for name in AUTOMATIC_DATASETS),
    conda:
        "../envs/shell.yaml"
    params:
        curl_args=CURL_ARGS,
        url=lambda wc: AUTOMATIC_DATASETS[wc.dataset]["url"],
    message:
        "Download stable dataset {wildcards.dataset}."
    shell:
        "curl {params.curl_args} --output {output:q} {params.url:q} > {log:q} 2>&1"


rule download_gst1:
    """Download the pinned GST-1 sediment-thickness archive from Zenodo."""
    output:
        archive=ensure(
            f"<resources>/automatic/{internal['resources']['automatic']['gst1']['filename']}",
            md5=internal["resources"]["automatic"]["gst1"]["md5"],
        ),
    log:
        "<logs>/automatic/download_gst1.log",
    conda:
        "../envs/shell.yaml"
    params:
        curl_args=CURL_ARGS,
        url=internal["resources"]["automatic"]["gst1"]["url"],
    message:
        "Download GST-1 sediment thickness from Zenodo."
    shell:
        "curl {params.curl_args} --output {output.archive:q} {params.url:q} > {log:q} 2>&1"
