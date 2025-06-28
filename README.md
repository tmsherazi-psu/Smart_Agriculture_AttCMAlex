# Breast_Cancer_Detection






### PROJECT OVERVIEW:
AI-Driven Smart Agriculture: Hybrid Transformer-CNN for Real-Time Disease Detection in Sustainable Farming




### IMAGE COUNTS:

data/
├── Train/
│   ├── cucumber/
│   │   ├── Healthy/
│   │   └── Sick/
│   ├── banana/
│   │   ├── Healthy/
│   │   ├── Sigatoka Disease/
│   │   └── Xanthomonas Infection/
│   └── tomato/
│       ├── Yellow Leaf Curl Virus/
│       ├── Bacterial Spot/
│       ├── Mosaic Virus/
│       ├── Late Blight/
│       ├── Septoria Leaf Spot/
│       ├── Spider Mites/
│       ├── Early Blight/
│       ├── Target Spot/
│       ├── Leaf Mold/
│       └── Healthy/
│
└── Validation/
    ├── cucumber/
    │   ├── Healthy/
    │   └── Sick/
    ├── banana/
    │   ├── Healthy/
    │   ├── Sigatoka Disease/
    │   └── Xanthomonas Infection/
    └── tomato/
        ├── Yellow Leaf Curl Virus/
        ├── Bacterial Spot/
        ├── Mosaic Virus/
        ├── Late Blight/
        ├── Septoria Leaf Spot/
        ├── Spider Mites/
        ├── Early Blight/
        ├── Target Spot/
        ├── Leaf Mold/
        └── Healthy/


### WHY EVOLUTIONARY ALGORITHMS:

Plant diseases pose a significant threat to global food security, with severe implications for agricultural productivity. Early and accurate detection of these diseases is crucial, yet it remains a challenging task, significantly impacting crop yields and food supply chains. 
Despite the progress in artificial intelligence, particularly deep learning, challenges persist in real-world applications due to environmental noise, varying light conditions, and other complicating factors that hinder detection accuracy. 
This study introduces the AttCM-Alex model, a novel deep-learning framework designed to boost the detection and classification of plant diseases under challenging environmental conditions. 
By integrating convolutional operations with self-attention mechanisms, AttCM-Alex effectively addresses the variability in light intensity and image noise, ensuring robust performance. 
To simulate practical agricultural scenarios, the study employs bilinear interpolation for image dimension adjustment and introduces Salt-and-Pepper noise. Additionally, the model’s robustness was evaluated by varying image brightness levels by ±10\%, ±20\%, and ±30\%.
Experimental results demonstrate that AttCM-Alex significantly outperforms traditional models, particularly in scenarios involving fluctuating light conditions and noise interference. The model achieved a peak detection accuracy of 0.97 with a 30\% increase in image brightness and maintained an accuracy of 0.93 even with a 30\% decrease in brightness, highlighting its robustness and reliability.
The findings affirm the AttCM-Alex model as a powerful tool for real-world agricultural applications, capable of enhancing disease detection systems' accuracy and efficiency. This advancement not only supports better crop management practices but also contributes to sustainable agriculture and global food security.