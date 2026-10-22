# Third-party materials and attribution

The repository's existing MIT license covers applicable project code; it does
not relicense upstream models, datasets, libraries or recovered third-party work.
Original research/recovery archives are excluded from this public source tree.
Retain upstream notices when redistributing any separately obtained materials.

| Material | Project use | Source/access |
|---|---|---|
| Llama 3.1 Instruct | Shared external-profile base | Obtain through the model publisher under its applicable terms |
| GPT-OSS-20B | Native MoE base | Obtain the matching upstream revision and terms |
| DeBERTa | Request classification | Matching upstream model and tokenizer required |
| GSM8K / SciQ / Dolly | Training/development/evaluation sources | Preparation scripts record source handling; obtain separately |
| PKU / ESConv / ToxicChat | Safety/support data sources | Research/reuse restrictions require source-specific review |
| PyTorch / Transformers / PEFT / vLLM / FastAPI | Runtime and experimentation | Installed separately using profile dependencies |

No model weights, raw conversations or held-out answer dumps are bundled.
Aggregate reports and hashes do not grant rights to underlying materials. This
inventory is not a completed legal clearance for future artifact redistribution;
review current upstream terms before publishing weights or raw examples.

The original project license attributes Ramanathan Swaminathan. Domain adapters
were recovered from earlier project artifacts and their provenance is not
represented as newly trained by this release. Authored runtime repairs,
subsequent experiments and source snapshots are distinguished in the reports.
