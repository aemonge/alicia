# Features

- [ ] Support Un tagged data: Cifar, EMNIST, CelebA, KMNIST, Omniglot, PhotoTour, Places265, SBU, STL10, USPS, VOC
- [ ] Implement a behavior to change hidden units on all models.
- [ ] Implement and extend with this other models.
    - [ ] SqueezeNet
    - [ ] Crazy (A mixed SqueezeNet with a really complex nn.Sequential)
    - [ ] VGG (AlexNet is fast; like how you would expect it)
    - [ ] Raw (Create it with nn.Sequential, and implement the math fn)
    - [ ] Choose from deep-ai.org
    - [ ] Try to get a open-ai model.
    - [ ] Inception
    - [ ] DenseNet
    - [ ] Alicia (a mixed model, from my favorites)
- [ ] Pay tech debt.
- [ ] Add an `-a, --auto [img-file]`  option to the `create` command to automatically set the input size
      based on the image resolution.
- [x] Add the network training history log, to the model. To enhance the info and comparer.

# Tech debt // Known Bugs

* [ ] Hide Elemental on production from `src/modules/models/__init__.py` import rule.
* [ ] Make pylint understand venv paths rightfully.
* [ ] Make pylint and pyright to understand the callback method class, or create a abstract class
* [ ] Change `--verbose` and display none, less, all information.
* [ ] Re-enable the console pure view (`plottext`)
* [ ] Choose a default width, and adjust all output to such a width
* [ ] `Tear_down`, `reset()` or `request.getfixturevalue()` fixture to avoid setting a manual order on pytest
