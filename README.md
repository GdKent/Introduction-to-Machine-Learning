# Introduction to Machine Learning: Course ISE - 364 / 464

![](Intro_to_ML.png)

Material for the course ISE-364/464, titled "Introduction to Machine Learning". This includes all original lecture slides that I created as well as homework assignments, projects, and the placement exam accompanying the course.

I originally built this course and taught it in the Fall 2024 semester at Lehigh University in the Industrial and Systems Engineering (ISE) Department. Two sections of this course were taught in tandem: a 364 section (for senior undergraduate students) and a 464 section (for graduate students).
As such, the lecture material remains the same for both sections, but the graduate section is simply augmented to have more difficult problems in the homework assignments and projects.

The material herein can be used in a variety of university, industrial, or recreational uses: to teach a university or industry-oriented introductory course on machine learning (ML), to serve as a comprehensive reference source for the fundamentals of the field of ML, or even topic-specific seminars (as the lectures are broken down by topic and are relatively independent from each other).

I will maintain this repository and keep the material updated if any edits are required. Further, I may develop more advanced material for future education in more specialized topics (deep learning, data mining, statistics, and / or optimization, etc.) which I will distinguish from the core course material as "special topics".

Please send any comments and corrections to me at gdk220@lehigh.edu. Further, if you are a course instructor seeking complete solution write-ups for the homework assignments, again, please email me at the aforementioned address.

## Course Overview, General Outline, & Prerequisite Knowledge

Machine learning (ML) is the study and development of algorithms that learn patterns from data in
an automated way and is the bedrock of the field of artificial intelligence (AI). This is an introductory course in ML
designed for senior undergraduate, master, and doctoral students who have a working knowledge of Python
and sufficient knowledge in probability, statistics, multivariable calculus, and linear algebra. This course introduces
the core principles of ML, fundamental techniques & models, data mining methodology, and prepares for more
advanced study in ML. Emphasis will be placed on introducing ML models in an intuitive way from the
fundamental mathematical building blocks in order to gain a deep understanding of the assumptions preceding each
algorithm. The learning of these concepts will be facilitated with homework assignments that will consist of a
mixture of mathematical and coding problems (emphasis will be placed on the mathematics behind the algorithms and experience applying these models in practice).

![](What_is_ML.PNG)

The general outline of topics that are covered in this course can be split up into the 6 following categories (by topic):

- **Introduction and Review Material:** Introuction to the topic of ML. Reviews of mathematics, linear algebra, multivariate calculus, probability, and statistics. Introduction to the fundamentals of numerical optimization.

- **Supervised Learning (Discriminative Models):** Topics span linear regression, logistic regression, K-nearest neighbors, decision trees, ensemble methods, support vector machines (SVM), and artificial neural networks.

- **Supervised Learning (Generative Models):** Topics span Gaussian discriminative analysis (GDA) and naive Bayes.

- **Unsupervised Learning:** Topics span K-means clustering and Gaussian mixture models.

- **Dimensionality Reduction:** Topics span principal component analysis (PCA) and linear discriminant analysis (LDA).

- **Data Mining Fundamentals:** Topics span the general procedure and workflow of the entire data mining process. This includes feature engineering, data cleaning, handling different types of data, feature encoding, model selection, cross-validation, and hyperparameter tuning.

A rough timeframe to expect the material to cover in the order listed is as follows: (1/2 - 1 months) Intro and Review Material, (1 - 1.5 months) Discriminative Supervised Learning, (1 - 2 weeks) Generative Supervised Learning, (1 - 2 weeks) Unsupervised Learning, (1 - 2 weeks) Dimensionality Reduction, and (1 - 2 weeks) Data Mining Fundamentals. The amount of time dedicated to each topic can naturally be modified to cater to the goal of the instructor as well as the needs of the class (for example, taking more time in covering the review material if the class consists of mostly undergraduate students).

Further, more advanced topics that may not fit within the original span of material covered by this course, but which is material that has been included in this repo (or will be uploaded in the future), and can be incorporated depending on the goals of the course are:

- **Deep Learning:** Topics could span convolutional neural networks, recurrent neural networks, long-short-term (LST) neural networks, generative adversarial networks (GANs), transformers (encoder & decoder architectures), and neural architecture search (NAS).

- **Advanced Unsupervised Learning:** Topics could span DBSCAN (density-based spatial clustering of applications with noise) and spectral clustering.

- **Advanced Dimensionality Reduction:** Topics could span T-SNE (t-distributed stochastic neighbor embeddings) and UMAP (uniform manifold approximation & projection).

- **Reinforcement Learning:** Topics could span Markov decision processes, policies & value functions, dynamic programming, Monte Carlo methods, temporal difference methods, and Q-learning.

## Coding Competencies Overview

As this is an introductory machine learning course, it is customary to incorporate a healthy amount of coding material to facilitate understanding and maximize learning outcomes by hands-on practice with algorithmic implementation or data mining. Naturally, Python is a first choice and is the language that is utilized in this class to
aid in learning the material. Choosing Python as the language for this class has several benefits, some of which include: (1) Python is perhaps the most intuitive of the coding laguages as well as the most "required-to-know" languages, (2) access to powerful scientific computing, machine learning, and deep learning (DL) libraries (such as Numpy, Scikit-Learn, Scipy, and PyTorch), (3) students will get experience with the most-used libraries for data mining and machine learning, (4) Pytorch is perhaps the most-used DL library for doing research and implementing modern-day advanced (and custom) ML and DL models.

This course is structured in a way where students are not required to have prior experience coding with Python (though it would be beneficial) as the first month or so is designed to include introductory material for each of the fundamental pyhon libraries: Numpy, Pandas, Matplotlib, and Seaborn. However, students are expected to have some coding experience before taking this course. This will aid in making the process of picking up the Python syntax more streamlined an will help cut down on the likelihoo of students falling behind.

![](Coding_Competencies.PNG)




