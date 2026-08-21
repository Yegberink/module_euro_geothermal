"""Rules to download automatic resource files for geothermal."""

CURL_ARGS = "--fail --silent --show-error --location --retry 5 --retry-delay 5 --retry-all-errors --continue-at -"


rule download_stable_dataset:
    output:
        "<resources>/automatic/{dataset}",
    log:
        "<logs>/automatic/download_stable_dataset_{dataset}.log",
    wildcard_constraints:
        dataset="|".join(internal["resources"]["automatic"]["datasets"]),
    conda:
        "../envs/shell.yaml"
    params:
        curl_args=CURL_ARGS,
        url=lambda wc: internal["resources"]["automatic"]["url"].format(
            dataset=wc.dataset
        ),
    message:
        "Download stable dataset {wildcards.dataset}."
    shell:
        "curl {params.curl_args} --output {output:q} {params.url:q} 2> {log:q}"
