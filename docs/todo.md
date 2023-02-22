# Features

- [ ] Support Un tagged data: Cifar, EMNIST, CelebA, KMNIST, Omniglot, PhotoTour, Places265, SBU, STL10, USPS, VOC
- [ ] Implement a behavior to change hidden units on all models.
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
- [ ] Support pre-trained models, with weights settings.

# Tech debt // Known Bugs

* [ ] Fix size information on the models, it doesn't match at all.
* [ ] Hide Elemental on production from `src/modules/models/__init__.py` import rule.
* [x] Make pylint understand venv paths rightfully.
* [x] Make pylint and pyright to understand the callback method class, or create a abstract class
* [ ] Change `--verbose` and display none, less, all information.
* [ ] Choose a default width, and adjust all output to such a width
* [ ] `Tear_down`, `reset()` or `request.getfixturevalue()` fixture to avoid setting a manual order on pytest
* [ ] When showing the probability, it should add up to lees than 1.0. Check it out
