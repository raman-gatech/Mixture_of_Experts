# Restricted demo deployment

The real demo target is the exact calibrated Stage 2 GPT-OSS archive, with the
external profile preserved for rollback. Stage 3 is not promoted. See
`../reports/deployment.json` for the latest verified release-deployment state;
a historical qualification is not a current live service.

Use a finite ICE GPU allocation, normal coc-ice QoS, one H100, scratch-backed
outputs/caches and inspected node-local model staging. Preserve HOME and active
OnDemand/VS Code sessions. Existing site-specific scripts are templates for
operators with cluster access, not commands for running inference on a login node.

The release demo mode `MOE_RELEASE_DEMO=1` disables automatic recollection of
research evaluation suites and the long research qualification loop. It retains
authenticated engine/gateway probes, cancellation/queue/history checks and actual
desktop/mobile browser checks. Fresh cold-start evidence is recorded. Existing
two-hour load/restart/rollback evidence remains historical unless rerun.

Application and engine bind to loopback. Use an SSH tunnel to the allocated node
and enter a privately provisioned application token in the UI. Do not share the
engine credential or publish either token, tunnel socket paths or access settings.
The allocation expiry must be displayed in handoff instructions. No 24/7 or
public endpoint is promised; zero paid cloud provisioning remains the policy.

Before describing a deployment as verified, record: source/config/weight identity,
job ID, verification timestamp, readiness, authorized and unauthorized requests,
representative real output, cancellation/overload cleanup, UI checks and known
failed checks. A readiness check alone does not establish quality or safety.

The optional Docker/Compose files describe an external-profile NVIDIA runtime.
They have not established native GPT-OSS container compatibility or a deployed
GPU container. Validate on a compatible host before using that claim.

## Walkthrough checklist

Record architecture and scope first, show authentication without exposing its
value, run a representative math/science/general interaction, show profile/task
selection and cancellation, then explain the results table and rejected Stage 3
candidate. State known failures and temporary hosting. Avoid sensitive advice
examples and do not cherry-pick a few outputs as evidence of benchmark accuracy.

A walkthrough recording has not yet been produced. Browser screenshots and live
probe evidence, when present, are integration evidence rather than a narrated video.
