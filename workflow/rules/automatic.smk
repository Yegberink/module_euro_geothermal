"""Rules to download automatic resource files for geothermal."""


rule download_stable_dataset:
    output:
        ensure(
            "<resources>/automatic/{dataset}",
            md5=lambda wc: AUTOMATIC_DATASETS[wc.dataset]["md5"],
        ),
    log:
        "<logs>/automatic/download_stable_dataset_{dataset}.log",
    wildcard_constraints:
        dataset="|".join(re.escape(name) for name in AUTOMATIC_DATASETS) or "(?!)",
    conda:
        "../envs/module.yaml"
    params:
        curl_args=CURL_ARGS,
        url=lambda wc: AUTOMATIC_DATASETS[wc.dataset]["url"],
    message:
        "Download stable dataset {wildcards.dataset}."
    shell:
        "curl {params.curl_args} --output {output:q} {params.url:q} > {log:q} 2>&1"
