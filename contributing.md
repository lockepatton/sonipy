# Collaborations are Welcomed!

Anyone can code, and we're super excited about collaborations on this project.
If you're planning a change, please reach out via email or by making an issue!
I'm just generally stoked to hear where you find this code useful, and happy to
give or receive advice.

# Feedback

This project is a cross-over between github, python code development and BVI
(blind and/or visually impaired) accessibility. If you have feedback or ideas,
please reach out! Send Locke an email at locke.patton@cfa.harvard.edu, or open
an issue on `sonipy`!

# Contributing - some basics to get you started

[Here](https://guides.github.com/activities/hello-world/) is a github basics tutorial.

First you will want to fork `sonipy` which creates your own copy of the complete code on github that
won't update until you specifically make changes to it. This way you can make independent changes
to your version of the code, without affecting everyone. To fork, click the "fork" button
on the top right of the homepage of `sonipy`.

Next clone your copy of `sonipy`. This copies all the files to your local machine.
You will want to work from command line or terminal to run these commands.

``` bash
  git clone <https://github.com/lockepatton/sonipy>
  cd sonipy
  python setup.py install
```

Make local changes to your machine here. If you are adding new functions, please
add additional functions in the ./tests/ folder to make sure that your code can be
checked as the code continues to change.

When you are happy with your changes, you can commit them and push them from your local machine.
Make sure you've tested the code and checked to see if the code passes all tests. See the
tests folder [README](https://github.com/lockepatton/sonipy/blob/master/tests/README.md) for more information on how to test on your local machine.

Finally, you can push to your fork of the project and submit a pull request. At that point you're
waiting on us. We can suggest changes or improvements or merge changes directly.

At any point in this process, you can reach out - either as an issue on github,
via email, or [twitter](https://twitter.com/Astro_Locke). This was once brand new to us too.
