# Sepsis Prediction

This project demonstrates transforming ML models into RESTful APIs and Graph APIs using FastAPI library, enabling easy integration into diverse applications. Learn to build ML models, create RESTful APIs, and containerize the application with Docker. This bridges the gap between ML practitioners and software engineers using different tech stacks.
This repository contains data analysis, insights, and machine learning modelling for sepsis prediction including a dev folder for model developmemt in Jupyter notebooks, a src folder containing both client frontend and FastAPI backend. This is a full stack application with FastAPI Backend encompasing a RESTFul and a GraphQL APIs.

<a name="readme-top"></a>

## Overview

Sepsis, a life-threatening condition, arises from the body’s exaggerated response to infection. Swift intervention is essential to prevent tissue damage, organ failure, and mortality because septic shock can cause death in as little as 12 hours. Therefore, Prompt diagnosis and treatment significantly enhance survival rates for individuals with mild sepsis. Conversely, without timely intervention, advanced stages of sepsis often prove fatal. Even with optimal care, septic shock—the most critical phase of sepsis—has a mortality rate of 30% to 40%.

## Key Objectives

`Early Detection:` Predict which patients in the ICU are likely to develop sepsis, allowing healthcare providers to intervene early and improve patient outcomes.

`Resource Allocation:` Optimize the allocation of medical resources by focusing attention and care on high-risk patients.

`Reduce Healthcare Costs:` Early detection and treatment of sepsis can decrease hospital stays and reduce the need for intensive care, leading to substantial cost savings.

`Improve Operational Efficiency:` Automating sepsis prediction through advanced machine learning models reduces the burden on healthcare professionals, allowing them to focus on patient care.

## Framework

The CRoss Industry Standard Process for Data Mining (CRISP-DM).

## Data dictionary

1. **ID**: Unique Identifier of patient
2. **PRG(Plasma Glucose)**: Measurement of plasma glucose levels.
3. **PL(Blood Work Result 1)**: Blood work result in mu U/ml.
4. **PR(Blood Pressure)**: Measurement of blood pressure in mm Hg.
5. **SK(Blood Work Result 2)**: Blood work result in mm.
6. **TS(Blood Work Result 3)**: Blood work result in mu U/ml.
7. **M11(Body Mass Index)**: BMI calculated as weight in kg/(height in m)^2
8. **BD2(Blood Work Result 4)**: Blood work result in mu U/ml
9. **Age**: Age of the patient in years.
10. **Insurance**: Indicates whether the patient holds a valid insurance card.
11. **Sepsis**: Target variable indicating whether the patient will develop sepsis (Positive) or (Negative)

### FastAPI backend

[API](https://gabcares-sepsis-fastapi.hf.space/docs)

[FastAPI Image](https://hub.docker.com/r/gabcares/sepsis-fastapi)

#### Demo video
https://github.com/user-attachments/assets/a260f4c1-bed2-40c8-903f-717fb7474bb1

### Streamlit frontend application

[client](https://gabcares-sepsis-streamlit.hf.space/)

[Streamlit Image](https://hub.docker.com/r/gabcares/sepsis-streamlit)

#### Demo video
https://github.com/user-attachments/assets/6489d547-c493-4387-b288-c9217aea67b0


## Technologies Used

- Anaconda
- Streamlit
- Python
- Pandas
- Plotly
- Git
- Scipy
- Sklearn
- Adaboost
- Catboost
- Decision tree
- Kneighbors
- LGBM
- LogisticRegression
- RandomForest
- SVC
- XGBoost
- Joblib

## Installation

### Quick install

```bash
 pip install -r requirements.txt
```

### Recommended install

```bash
conda env create -f sepsis_environment.yml
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 💻 Getting Started

To get a local copy up and running, follow these steps.

### Prerequisites

- [Docker Desktop](https://docs.docker.com/desktop/)

### Setup

Clone this repository to your desired folder:

```sh
  cd your-folder
  git clone https://github.com/D0nG4667/sepsis_prediction_full_stack.git
```

Change into the cloned repository

```sh
  cd sepsis_prediction_full_stack
  
```

After cloning this repo,

- Add an env folder in the root of the project.

- Create and env file named `offline.env` using this sample

```env
# API
API_URL=http://api:7860/api/v1/prediction?model

# Redis local
REDIS_URL=redis://cache:6379/
REDIS_USERNAME=default
```

- Run these commands in the root of the repo to explore the frontend and backend application:

```sh
docker-compose pull

docker-compose build

docker-compose up

```

## Contributions

### How to Contribute

1. Fork the repository and clone it to your local machine.
2. Explore the Jupyter Notebooks and documentation.
3. Implement enhancements, fix bugs, or propose new features.
4. Submit a pull request with your changes, ensuring clear descriptions and documentation.
5. Participate in discussions, provide feedback, and collaborate with the community.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Feedback and Support

Feedback, suggestions, and contributions are welcome! Feel free to open an issue for bug reports, feature requests, or general inquiries. For additional support or questions, you can connect with me on [LinkedIn](https://www.linkedin.com/in/dr-gabriel-okundaye).

Link to article on Medium: [Building Machine Learning APIs with FastAPI: Embedding Machine Learning Models for Predicting Sepsis- A Full Stack Approach](https://medium.com/@gabriel007okuns/building-machine-learning-apis-with-fastapi-embedding-machine-learning-models-for-predicting-bd8ed66efc6b)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 👥 Authors <a name="authors"></a>

🕺🏻**Gabriel Okundaye**

- GitHub: [GitHub Profile](https://github.com/D0nG4667)

- LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/dr-gabriel-okundaye)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## ⭐️ Show your support <a name="support"></a>

If you like this project kindly show some love, give it a 🌟 **STAR** 🌟. Thank you!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📝 License <a name="license"></a>

This project is [MIT](/LICENSE) licensed.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
