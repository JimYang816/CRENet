"""Authenticate checkpoints made immediately before the CRENet rename.

The two identities bind the old source and its renamed equivalent. This does
not relax data-view checks or accept checkpoints from different code versions.
"""

LEGACY_SCHEMA = "watermark-icr-mmse-net-view-band14-v3"

LEGACY_INFERENCE_CODE = {
    "data_generation.src.dataset.compact_math:reconstruct": "5d34c1ae871783807cadbdba21d9cd489751e4f80b387fe4f84bf5dcfc765aec",
    "data_generation.src.receiver.dpss:DPSSModel": "e7bd79721a946c29a5a5ae0412e152bfc5eb03d7b579b22a8a0d48475168735e",
    "src.algorithms.icr_mmse.checkpoint": "9d70c284f00ebf9811603220c497b2441db5c57370f8cb12a67a743f7fbda079",
    "src.algorithms.icr_mmse.dpss": "b050e0d0742a9902ed64a8a91c066eb2800e5545ff5d351148c5f4d789d01177",
    "src.algorithms.icr_mmse.lmmse": "7e46891d3641702f78a879e1103fc5911a3410bac27da48239d8a56d929ff4a1",
    "src.algorithms.icr_mmse.model": "fbeb14cf2ce96d1fc0a2500215e5a94469b66afd77927e0e7c368b595e0f1ce5",
    "src.algorithms.icr_mmse.operators": "f77b11dca416a26f8f901ac90a2bd261a87923e7f251e8a8cac1f76bb7b479b9",
    "src.common.numerics": "0c99f022856c9db8f0355a898bd3eda35a1a569398ab573be0c78c0b5476ea6f",
    "src.data.compact:CompactDataset": "28e544c2bbc7573ebe76a68e4b6f0308694271956528398bf8854a81222e62b3",
    "src.data.contract": "fa7e36c5b216e82b450c90844d985b4f8db948399354eb0705f099007b7964ed",
    "src.data.icr_mmse": "c480df7667765865f4c711d2fff927447df6b02d1b0a85c2a7faad473d9e582d",
    "src.evaluation.inputs": "f63f75dbbc2c69324d9828bb31f57291c959acb04526f9e18e6447d5b1fe3bc8",
}

RENAMED_INFERENCE_CODE = {
    "data_generation.src.dataset.compact_math:reconstruct": "5d34c1ae871783807cadbdba21d9cd489751e4f80b387fe4f84bf5dcfc765aec",
    "data_generation.src.receiver.dpss:DPSSModel": "e7bd79721a946c29a5a5ae0412e152bfc5eb03d7b579b22a8a0d48475168735e",
    "src.algorithms.crenet.checkpoint": "ee34bd0cc374297dff57106d000a4e30b5777c48613a04b124ba37be04420224",
    "src.algorithms.crenet.dpss": "b050e0d0742a9902ed64a8a91c066eb2800e5545ff5d351148c5f4d789d01177",
    "src.algorithms.crenet.lmmse": "7e46891d3641702f78a879e1103fc5911a3410bac27da48239d8a56d929ff4a1",
    "src.algorithms.crenet.model": "9a1e2ad30663e5a149114f21f84654ca0183389d1ea2b7ac1d59ce7cec78b096",
    "src.algorithms.crenet.operators": "f77b11dca416a26f8f901ac90a2bd261a87923e7f251e8a8cac1f76bb7b479b9",
    "src.common.numerics": "0c99f022856c9db8f0355a898bd3eda35a1a569398ab573be0c78c0b5476ea6f",
    "src.data.compact:CompactDataset": "28e544c2bbc7573ebe76a68e4b6f0308694271956528398bf8854a81222e62b3",
    "src.data.contract": "fa7e36c5b216e82b450c90844d985b4f8db948399354eb0705f099007b7964ed",
    "src.data.crenet": "4efd56c3c4aa071b698f4513ec56f242c7c6e582c5db7b7442edae7a62591c10",
    "src.evaluation.inputs": "f63f75dbbc2c69324d9828bb31f57291c959acb04526f9e18e6447d5b1fe3bc8",
}

# The formal 2026-10-31 weights predate 027ae0e, whose only numerics.py
# change adds default_device(). DenseBackend and all numerical operators are
# identical. Bind this second complete historical identity explicitly; never
# ignore a mismatched hash or rewrite the checkpoint on disk.
FORMAL_INFERENCE_CODE = dict(LEGACY_INFERENCE_CODE, **{
    "src.common.numerics": "c43dc2516064c33be1b209c3edf521e33ce51116465fdfafbbcea2b26cf04690"})


def matches_renamed_checkpoint(payload, current_code):
    return (payload.get("schema") == LEGACY_SCHEMA
            and payload.get("inference_code") in (LEGACY_INFERENCE_CODE, FORMAL_INFERENCE_CODE)
            and current_code == RENAMED_INFERENCE_CODE)
