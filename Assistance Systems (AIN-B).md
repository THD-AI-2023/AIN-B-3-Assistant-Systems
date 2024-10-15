# Assistance Systems (AIN-B)

Recommendation System Project

Prof. Dr.-Ing. Udo Garmann

### Abstract

Requests for the project, need to be specified more by the students!

## Introduction

Here are the requests for the "Assistance Systems” project.

Please notice that, in Software Engineering, the term 'requests' refers to first coarse descriptions beforethe specifications / requirements. So it is one of your tasks during the project development to specify details according to the topics learned in the lectures.

Also, the grading is not about just “check marking" the requests, but how good they have been further engineered and implemented.

## Project Setup in MyGit

Login to remote MyGit and create a project, that will be used as the basis for grading of the course.

If not done yet, setup the local Development Environment (VS Code, PyCharm or similar) with GIT and Python.

Please add me (Prof. Garmann) as a member (or developer) to your MyGit project, so that I can see the files (Guest status is not enough for this!).

## Use of MyGit

The following chapter 'README structure' describes the students and the project.

The Wiki on MyGit is the detailed documentation of the project.

The repository stores the raw files in the folder structure of streamlit and rasa. The files must not be zipped or otherwise compressed.

No additional files must be added, especially no model-files!

## README structure

By the end of the semester there will be an iLearn slot to submit a README.md file for the project with a deadline.

The README.md must have the specific structure:

- The first line must contain the lastname, firstname and the mat-no of the first student(e.g.Garmann,Udo, 123456)

- then an empty line must be included

-  lastname, firstname and the mat-no of the second student then an empty line must be included. 

- The title name of the project 

- then an empty line must be included

- A link to the MyGit Repository

- then an empty line must be included

- A link to the MyGit Wiki

-  A MD chapter called "Project description". In this chapter you describe the basics of the project.

-  A MD chapter "Installation", which describes the prerequisites and installation of the project on another computer. Especially declare the used versions (of Python,scikit-learn,streamlit,rasa,...) here.

- A MD chapter "Data" must be added. It must contain a link to the original data source, the approach for handling outliers and the one for creating fake data.

- A MD chapter "Basic Usage". It describes how to start the project, first logins with passwords and key use cases.

- Later, a MD chapter "Implementation of the Requests" must be added. It describes how the code implements a request. Also it must describe, how the a student contributed to the implementation of the request.

- Add a MD chapter for arguing about the "right-fit question" for a chatbot.

- Also, a MD chapter "Work done" must be added later. It must describe, who has implemented which request.

-  You may add additional chapters for more details about the project.

## Part 02 Requests

Create a data-driven web application that implements the following requests:

1. A multi-page Web App with Streamlit (https://streamlit.io) has to be developed.

2. For source code and documentation, a MyGit repository and Wiki has to be used.

3. Develop and describe 3 Personas in the Wiki.

4. Find 5 use cases for the application.

5. A requirements.txt file must be used to list the used Python modules, which can be used to install them.

6. A README.md file must be created with the structure described in part 01.

7. The module venv for a virtual project environment must be used.

8. A free data source must be used. You may find it for example at Kaggle, SciKit (but not the built-in ones), or other.

9. There must be a data import (predefined format and content of CSV).

10. The data must be analyzed in the app (e.g. with Pandas, a Jupyter notebook must not be submitted),so that an app user gets on overview (e.g. of correlations, min/max, median,..). The result must be visualized in the app.

11. Identify and update outliers, if some exists. Explain your approach in the data chapter of the Wiki.

12. The data must be transformed so that it can be used in the app. Follow the descriptions in https://developers.google.com/machine-learning/crash-course/numerical-data and following chapters about data.

13. Add 25-50% realistic fake data sets. Explain you approach in the data chapter of the Wiki. What is the effect to the training of the model.

14. Create several input widgets (at least 3, where 2 must be different) that change some feature variables.15. At least 2 Scikit-Learn model training algorithms (e.g. from Aurélien Géron, Chapter 4) must be applied in order to predict some variable(s). Argue in the Wiki about which one is best suited for the app.

16. Select a use case for which the "right fit" question for chatbots has a positive answer. Argue about why using a chatbot fo the feature makes sense.

17.Create sample dialogs for the use case. Document them in the Wiki.

18. Create a high-level (dialog) flow for the use case. Also document it in the Wiki.

19. Create a rasa chat bot that must be included for a use case. Add the source files to the MyGit repository.

20. Create a video/screencast of your project. The video must show at least 3 use cases, one of them is the rasa chatbot.

21. This list of requests may be updated throughout the semester. You will be informed about this in lectures and/or mail-messages.

## Grading

In general, the project about a recommendation system has the fllowing parts:

1) Graphical User Interface (GUI)

2) Visualization (with pandas and matplotlib)

3) Data analysis with pandas

4) Outliers and fake data

5) Scikit-learn

6) Sample-dialogs

7) Dialog flow

8) rasa implementation

When the project is done by 2 students, they mst divide their work like this:

Student 1:

1) Graphical User Interface / Visualization

2) General Data analysis

3) Sample dialogs

Other student:

4) Strategies for outliers and fake data

5) Scikit-Learn

6) Dialog flow

Both: 7) Documentation and Programming.

There will be the folowing parts of grading:

1) Formal aspects (submission , how descriptions are followed, _.) (12,5%).

2) How good the requests are implemented (50%).

3) General Impression (12,5%).

4) Implementation principles (25%).



