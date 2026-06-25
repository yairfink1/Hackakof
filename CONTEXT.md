We are the CTOs for a startup, choosing between two options:

A shared bike company.

Pixel Perfect—robust object detection in an image.

There is time until Friday at 11:00 AM to submit the results, the code, and the instructions on Moodle.

They want to see creativity.

There is no "textbook solution"; the goal is to do your best to solve the problem.

You can search for additional data online.

There are forums on Moodle for questions.

Suggested Time Allocation:
Choose a project to work on, and define when you start and finish working.

Keep a 5-hour block saved for submitting the assignment, so you don't end up trying to submit at the very last minute.

Strive as quickly as possible for something submittable—MVP: Aim for the simplest model first, one that is good enough to submit.

Iterations—clean the data, try more complex models, hyperparameter tuning!

Tips:
Decide on a data split. For example, start by working with only 50% of the data, split it into train/test, and add more data over time.

Use git and docopt.

Pay attention to the notes on Moodle!

Read the instructions carefully at the beginning! Work in parallel!

Use GEMINI!!! (Yessssss) antigravity also works, you just can't copy anyone else's code! (And you also need to understand the code in the end).

You can do anything except copy work that someone else did before.

Choose the task that looks the most interesting to you.

Talk to other people about what they are doing.

Don't get stuck—make reasonable assumptions and move on.

Task Description:
Image Recognition:
You need to detect the monkey in different environments, not just in the same environment—meaning the model needs to understand that what it is learning is just the monkey, not its background. Ensure an identical distribution of backgrounds in the train and test sets. The model must be robust. They aren't telling us what they will do to the data; they might turn it black and white, flip it, warp it, and our model needs to be robust and succeed!
The metric is accuracy.

General Notes:
Don't be tempted to go straight for the best model right at the start. Instead, choose a model each time, train it until it reaches a plateau, then test another model, train and improve it, to ultimately reach the best result.

The interview is also part of the grade (2 points). You don't need to show that you know exactly every single line of code, but rather show that you tried different things, what decisions you made, and that you applied things learned in the course.

There is a check_submission file—test it right at the beginning with the baseline.

There is also an evaluation file—use it to see how well our model performs according to their ranking.

It will be best for me to run on a GPU, so im trying to work in git and run in googlde cloud.