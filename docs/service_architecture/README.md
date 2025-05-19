# MUSE: User-Sensitive Experience (E6692 Spring 2025: Final Project)
MUSE is a system designed to provide active, low-latency audio stimulation aimed at enhancing motivation, focus, and mood stabilization. By fusing real-time activity data with stimulatory audio generation, MUSE seeks to improve focus and task performance.


#### Service Architecture
The system streams mobile data to a systematically finetuned data encoder, linked to a pretrained audio decoder within a joint embedding (semantic latent-space) architecture.
Generatede streams are sent to the user's edge device to be processed into a smooth musical experience

## Repository Organization

The repository is structured as follows:

```
e6692-2025spring-FinalProject-MUSE-lm3963/
├── data/                   # Contains datasets used for training and evaluation
├── models/                 # Pretrained models and model checkpoints
│   ├── encoder/            # Activity-sensitive mobile-data encoder
│   ├── decoder/            # Pretrained audio decoder
│   └── stream_prcs/        # Real-time data streaming and edge processing
├── src/                    # Source code for the project
|   ├── edge_acq/           # Data Acquisition on edge device (Apple Watch)
|   ├── edge_strm/          # Streaming processes on edge device (Jetson/iphone)
│   ├── VM_prcs/            # Model architecture and training scripts
│   |   └── data_prcs/      # Scripts for data preprocessing and augmentation
│   ├── evals/              # Scripts for evaluating model performance
│   └── utils/              # Utility functions and helper scripts
├── scratch/                # Development, Experiments and visualization files
├── docs/                   # Documentation and project-related resources
│   └── service_archtr/     # Service architecture & runtime docs
├── results/                # Generated results, logs, and analysis
├── tests/                  # Unit tests for the project
├── requirements.txt        # High-level dependencies
└── README.md               # Project overview and instructions
```

### How to Run (offline mode)
1. Clone the repository:
    ```bash
    git clone https://github.com/username/e6692-2025spring-FinalProject-MUSE-lm3963.git
    cd e6692-2025spring-FinalProject-MUSE-lm3963
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the main script:
    ```bash
    python src/main.py
    ```
4. For detailed instructions, refer to the [docs/](docs/) directory.


## Instructions for E6692 README.md file structure

The description of the final project for e6692 2025 spring is provided in [the Google drive](https://docs.google.com/document/d/1ysuf-gNWOS9CF6A7tQjX72crqGVqCAQJkAoN8FW9xHg/edit?usp=drive_link)

Students need to maintain this repo to have a "professional look":
* Remove the instructions (this text)
* Provide description of the topic/project
* Provide organization of this repo 
* Add all relevant links: name of Google docs and link to them, links to public repos used, etc.
* For paper reviews it should include the organization of the directory, brief description how to run the code, what is in the code, links to relevant papers, links to relevant githubs, etc...
