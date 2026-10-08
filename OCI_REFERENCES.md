# OCI references for ReviewOps delivery

Mode: Kubernetes manifests, Podman image publication, and user-run OKE access instructions. No OCI SDK/API/Terraform code or new cluster stack is generated.

User-selected CLI profile: `axon`, set with `export OCI_CLI_PROFILE=axon`. Intended kubeconfig context: `context-cef-oke-liftoff-ch4yy43wxtq`. User terminal results confirm cluster access, an AMD64 application rollout from OCIR, the public LB, and a Ready KServe predictor with passing smoke checks. These cloud results were supplied by the user. No OCI-affecting commands have been executed.

Local container runtime: Podman 5.6.2 on an Arm64 Mac; the AMD64 OKE image is built and tested under emulation, and Arm64 remains for local testing. The official CPU package index confirmed torch 2.14.1+cpu availability; the actual Linux build and inference checks passed. The Python base digest and image/dependency evidence are recorded in the Dockerfile and `artifacts/container/`. Latest OCI CLI version was not checked; no version-specific OCI CLI operation is generated.

Sources checked:

- [OKE private image pulls](https://docs.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengpullingimagesfromocir.htm)
- [OCIR naming and concepts](https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryconcepts.htm)
- [OCIR image publication](https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registryimagespushingpulling.htm)
- [OKE cluster access](https://docs.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengdownloadkubeconfigfile.htm)
- [Kubernetes probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
- [Official PyTorch CPU installation](https://docs.pytorch.org/get-started/locally/)

The [runbook](docs/reviewops-oke-runbook.md) separates local validation, user-run read-only cloud checks, registry publication, and mutating Kubernetes apply steps. Cloud instructions are templates until a repository and matching worker architecture are confirmed.

Push handoff refreshed 2026-10-06 against official OCIR repository-creation and image-push instructions. Uses local Podman 5.6.2 tag/login/push with authfile and digestfile; no OCI CLI API operation is generated or executed.

- [Create an OCIR repository](https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrycreatingarepository.htm)
- [OCIR login and image push](https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrypushingimagesusingthedockercli.htm)

Public-LB handoff: user-run Kubernetes Service `reviewops-public`, port80 to named app port `http`, OCI flexible LB annotations (10 Mbps minimum/maximum). No cloud action executed. Existing ClusterIP Service retained.
- [OCI LB provisioning annotations](https://docs.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengcreatingloadbalancers-subtopic.htm)

KServe installation handoff checked 2026-10-06: user-reported Kubernetes1.35.2; cert-manager pods ready and Certificate CRD present. KServe v0.20.0 Helm CRD/resources charts, explicit Standard mode, disableIngressCreation=true, enableGatewayApi=false; predictor will use internal Service behind existing FastAPI/OCI LB. These are user-run commands; controller installation alone does not migrate application inference.
- [KServe Standard installation](https://kserve.github.io/website/docs/admin-guide/kubernetes-deployment)
- [Pinned v0.20.0 chart values](https://github.com/kserve/kserve/blob/v0.20.0/charts/kserve-resources/values.yaml)
- [KServe ingress controls](https://kserve.github.io/website/docs/admin-guide/configurations)

KServe controller pull diagnosis: user-provided events show successful scheduling, ready webhook Certificate, successful RBAC-proxy image pull, and controller ErrImagePull/ImagePullBackOff due to enforcing short-name ambiguity. Proposed minimal Helm correction: kserve.controller.image=docker.io/kserve/kserve-controller, retaining v0.20.0 and existing Standard/internal-only values. Not yet verified on cluster; user-run upgrade and readiness checks required. Other optional chart/runtime images must also use fully qualified registries when those components are enabled.

KServe custom-predictor manifest prepared after user-reported controller rollout2/2 Running. k8s/kserve/reviewops-sentiment.yaml reuses published AMD64 digest and existing reviewops/ocir-pull secret. Standard, cluster-local, one replica, autoscaler disabled for first validation; no external predictor ingress. Existing /predict API retained (not an Open Inference Protocol compliance claim). Original browser app still performs local inference until a separate frontend delegation change. Local YAML parse passed; server dry-run, apply, readiness, and predictor inference remain user-run/unverified.
- [KServe custom predictor schema v0.20.0](https://github.com/kserve/kserve/blob/v0.20.0/pkg/apis/serving/v1beta1/predictor.go)
- [KServe cluster-local and autoscaler configuration](https://kserve.github.io/website/docs/admin-guide/configurations)

Frontend delegation completed locally on 2026-10-06: 47 tests pass; final AMD64 frontend image passes real remote predictions, no frontend weight loading, outage503, and recovery checks. The new frontend image is not published or deployed. The user-run guide uses the existing OCIR repository, pull secret, Deployment, and LB, with one atomic image/backend-environment update. See docs/reviewops-kserve-frontend.md. No OCI-affecting command was executed. Earlier unverified handoff statements above describe their historical preparation stages; the user subsequently verified the controller and predictor.

2026-10-06 subsequent user verification: public smoke test against 129.80.162.140 passed health, readiness, UI, positive/negative predictions, input limits, model revision, and inference-backend checks. Both predictions returned inference_backend=kserve and pinned hf:714eb0fa89d2f80546fda750413ed43d93601a13. Predictor compute times reported21.76ms/21.17ms; these are not end-to-end latency measurements. This supersedes the earlier cloud-pending status. Actual running frontend image digest was not supplied in this message.
