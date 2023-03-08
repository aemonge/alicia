# Features

- [-] Support Un tagged data
    - [x] CIFAR10
    - [x] EMNIST
    - [x] StanfordCars
    - [x] KMNIST
    - [-] Omniglot: Missing category labels
    - [-] Places265: Error 404
    - [-] Kitti: File to big for my /tmp
    - [-] EuroSAT: Requires Secure Certificate
- [x] Implement a behavior to change hidden units on all models.
- [ ] Merge create with modify.
- [ ] Implement and extend with this other models.
    - [x] SqueezeNet
    - [x] AlexNet
    - [x] MNASNet
    - [ ] Choose from deep-ai.org
    - [ ] Try to get a open-ai model.
    - [ ] Inception3
    - [ ] Alicia (a mixed model, from my favorites)
- [ ] Pay tech debt.
- [x] Add an `-a, --auto [img-file]`  option to the `create` command to automatically set the input size
      based on the image resolution.
- [x] Add the network training history log, to the model. To enhance the info and compare.
- [x] Support pre-trained models, with weights settings.

# Tech debt // Known Bugs

* [ ] When showing the probability, it should add up to lees than 1.0. Check it out
* [ ] Fix size information on the models, it doesn't match at all.
* [ ] Remove the UnShapetransform and Use the Reshapetransform((1,28,8)) instead. Check with unit test
* [ ] Centralize **all** dependencies, in the dependencies main package, beware of circular dependencies.
      (https://pypi.org/project/dependency-injector/)
* [ ] Hide Elemental on production from `src/modules/models/__init__.py` import rule.
* [ ] Fix all pylint, pyright and robocop issues
* [ ] `Tear_down`, `reset()` or `request.getfixturevalue()` fixture to avoid setting a manual order on pytest
* [ ] Processing huge data, kills the app. Make it performant.
* [ ] Change `--verbose` and display none, less, all information.
* [ ] Choose a default width, and adjust all output to such a width

# Extras (from new learnings from udacity-deeplearinig)

* [ ] Use the `Softmax` function as an output, since it will return the probabilities for our classes.
* [ ] Implement a random start for the trainer, to avoid local getting stuck at mins
* [x] Use a network with Conv2d to avoid network coupled with the image sizes, use the `final-mnist.qmd` as an example.
* [x] Save the model path location on itself to get the right size
* [x] Save the last image transform on the model to simplify the usage
* [x] Also save the data, and data/labels in the pth. With an option on the create/modify to "trim", but they're simple string should be ok
