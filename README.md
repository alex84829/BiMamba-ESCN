# BiMamba-ESCN
BiMamba-ESCN: Topology-Preserving Bidirectional State-Space Modeling for EEG-Based Assessment of Disorders of Consciousness

## Introduction <a name="Abstract"></a>
Accurate assessment of disorders of consciousness (DOC), particularly distinguishing between unresponsive wakefulness syndrome (UWS) and minimally conscious state (MCS), remains challenging. We propose BiMamba-ESCN, a topology-preserving EEG framework that integrates grouped cross-attention (GCA) and gated axial state-space modeling (GASS) to jointly capture spectral–connectivity interactions and spatiotemporal dependencies. Under a strict subject-wise, leakage-free evaluation protocol, the proposed method demonstrates robust and clinically meaningful performance for EEG-based DOC assessment.

## Overall Pipeline
<img width="4247" height="932" alt="overall_picture" src="https://github.com/user-attachments/assets/0e686c37-c8cc-4d49-87cd-93717a0eb470" />

Model Components

1.GCA: Topology-preserving cross-attention for structured fusion of spectral (PSD/DE) and connectivity (PLV/wPLI) features.

2.GASS: Bidirectional state-space modeling along channel and temporal axes with adaptive gating.

3.Topology Design: Maintains channel–frequency structure throughout the pipeline for physiologically meaningful representation.

## Usage

Installation
1) Clone the repository:<br />
```git clone https://github.com/alex84829/BiMamba-ESCN.git``` <br /><br />

2) Install the required dependencies<sup></sup>:<br />
```pip install -r BiMamba-ESCN/requirements.txt```

Run training

Run the following command to train the model:

```bash
python train.py
```

## Citation <a name="citation"></a>
"BiMamba-ESCN: Topology-Preserving Bidirectional State-Space Modeling for EEG-Based Assessment of Disorders of Consciousness" has been submitted to the journal Transactions on Biomedical Engineering.

@misc{BiMamba-ESCN,

  title={BiMamba-ESCN: Topology-Preserving Bidirectional State-Space Modeling for EEG-Based Assessment of Disorders of Consciousness},

author={Feng Gao, Linbo Qing, Lindong Li, Li Gao},

  journal={IEEE Transactions on Biomedical Engineering},

}
