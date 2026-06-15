# BiMamba-ESCN
Topology-preserving bidirectional state-space model (BiMamba-ESCN) for EEG-based assessment of disorders of consciousness (UWS vs MCS)

## Introduction <a name="Abstract"></a>
Accurate assessment of disorders of consciousness (DOC), particularly distinguishing between unresponsive wakefulness syndrome (UWS) and minimally conscious state (MCS), remains challenging. We propose BiMamba-ESCN, a topology-preserving EEG framework that integrates grouped cross-attention (GCA) and gated axial state-space modeling (GASS) to jointly capture spectral–connectivity interactions and spatiotemporal dependencies. Under a strict subject-wise, leakage-free evaluation protocol, the proposed method demonstrates robust and clinically meaningful performance for EEG-based DOC assessment.

## Overall Pipeline
<img width="4247" height="932" alt="overall_picture" src="https://github.com/user-attachments/assets/0e686c37-c8cc-4d49-87cd-93717a0eb470" />

Model Components
1)GCA: Topology-preserving cross-attention for structured fusion of spectral (PSD/DE) and connectivity (PLV/wPLI) features.

2)GASS: Bidirectional state-space modeling along channel and temporal axes with adaptive gating.

3)Topology Design: Maintains channel–frequency structure throughout the pipeline for physiologically meaningful representation.
