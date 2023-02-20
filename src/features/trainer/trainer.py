from dependencies.core import *
from dependencies.datatypes import *
from libs import PrettyTrain, UnLabeledImageDataset

class Trainer(PrettyTrain):
  """
    Base class for all training

    Attributes
    ----------
      model: AbsModule
        The model to train.
      Transforms: Transforms
        The transformations to apply to the data.
      learning_rate: float
        The learning rate to use.
      momentum: float
        The momentum to use.

    Methods
    -------
      train_step(images: torch.Tensor, labels: torch.Tensor, loss_count: float) -> float
      validation_step(dataloaders: DataLoader, batch_size: int) -> tuple[float, int]
      train(self, data_dir: str, labels: dict, batch_size: int = 64, epochs: int = 1,
            freeze_parameters: bool = False) -> None
      test(self, data_dir: str, labels: dict, batch_size: int = 64, freeze_parameters: bool = False)
      predict(self, image: Pil.Image, topk:int=5)
  """

  def __init__(
    self, model: AbsModule, transforms, learning_rate: float = 1/137,  momentum: float|None = None
  ) -> None:
    """
      Parameters
      ----------
        model: AbsModule
          The model to train.
        transforms: Transforms
          The transformations to apply to the data.
        learning_rate: float
          The learning rate to use.
        momentum: float
          The momentum to use.

      Returns
      -------
        None
    """
    super().__init__()
    self.model = model
    self.learning_rate = learning_rate

    self.criterion = torch.nn.CrossEntropyLoss()
    if momentum is not None:
      self.momentum = momentum
      self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
    else:
      self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
    self.transforms = transforms

  def train_step(self, images: torch.Tensor, labels: torch.Tensor, loss_count: float) -> float:
    """
      Performs a single training step.

      Parameters
      ----------
        images: torch.Tensor
          The images to train on.
        labels: torch.Tensor
          The labels to train with.
        loss_count: float
          The loss count to increment.

      Returns
      -------
        : float
          The loss count.
    """
    self._spin()

    self.optimizer.zero_grad()
    self._spin()

    output = self.model(images)
    self._spin()

    loss = self.criterion(output, labels)
    self._spin()

    loss.backward()
    self._spin()

    self.optimizer.step()
    self._spin()

    loss_count += loss.item()
    self._spin()

    return loss_count

  def validation_step(self, dataloaders: DataLoader, batch_size: int) -> tuple[float, int]:
    """
      Performs a single validation step.

      Parameters
      ----------
        dataloaders: DataLoader
          The dataloaders to use.
        batch_size: int
          The batch size to use.

      Returns
      -------
        : tuple[float, int]
          The loss count and the number of correct predictions.
    """
    ix = 0
    vd_loss = 0.0
    vd_correct = 0
    validate_loader_count = len(dataloaders.dataset) # pyright: reportGeneralTypeIssues=false

    with torch.no_grad():
      self.model.eval()
      self._spin(step='valid')

      for (images, (labels, _)) in iter(dataloaders):
        ix += batch_size * 1
        self._spin(step='valid')

        output = self.model.forward(images)
        self._spin(step='valid')

        loss = self.criterion(output, labels)
        self._spin(step='valid')

        vd_loss += loss.item()
        self._spin(step='valid')

        ps = torch.exp(output)
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        vd_correct += equals.sum().item()
        self._spin(step='valid')

        self._loading(ix, validate_loader_count, step = 'valid')

    self._spin(step='valid')
    self.model.train()
    return vd_loss, vd_correct

  def train(self, data_dir: str, labels: dict, batch_size: int = 64, epochs: int = 1,
            freeze_parameters: bool = False) -> None:
    """
      Trains the model.

      Parameters
      ----------
        data_dir: str
          The path to the data directory.
        labels: dict
          The labels to train with.
        batch_size: int
          The batch size to use.
        epochs: int
          The number of epochs to train for.
        freeze_parameters: bool
          Whether to freeze the parameters.

      Returns
      -------
        None

      Help:
      ----------
        labels['25.jpg'] # -> Str
        category_labels_ids[labels['25.jpg']] # -> Int
    """
    if 'train' not in self.transforms.keys() or 'valid' not in self.transforms.keys():
      raise ValueError('Transforms must be defined and set')

    category_labels_ids = { v:k for k,v in enumerate(self.model.labels)}
    train_ldr = DataLoader(UnLabeledImageDataset(
        f"{data_dir}/train", labels, category_labels_ids, transform = self.transforms['train']
      ), batch_size = batch_size, shuffle=True
    )
    valid_ldr = DataLoader(UnLabeledImageDataset(
        f"{data_dir}/valid", labels, category_labels_ids, transform = self.transforms['valid']
      ), batch_size = batch_size, shuffle=True
    )

    train_loader_count = len(train_ldr.dataset)
    validate_loader_count = len(valid_ldr.dataset)

    self._print_train_header(epochs, batch_size, train_loader_count, validate_loader_count)

    time_count = 0
    vd_correct = 0
    total_time = time.time()
    start_time = total_time

    # Freeze parameters
    if (freeze_parameters):
      for param in self.model.features.parameters():
          param.requires_grad = False

    self.model.train()
    for epoch in range(epochs):
      tr_loss = 0.0
      ix = 0

      self._print_step_header(epochs, epoch)
      for (images, (labels, _)) in iter(train_ldr):
        ix += batch_size * 1
        tr_loss = self.train_step(images, labels, tr_loss)
        self._loading(ix, train_loader_count)
        if math.isnan(tr_loss):
          raise Exception('Loss has been lost, check parameters')

      else:
        vd_loss, vd_correct = self.validation_step(valid_ldr, batch_size)

        self.__backspace__(hard=True)
        time_now = self._print_step(
          epoch, epochs, start_time, time_count,
          tr_loss, vd_loss, vd_correct,
          validate_loader_count, train_loader_count
        )
        start_time = time_now
    else:
      self._print_total(vd_correct, validate_loader_count, total_time)

    self.model.training_history.append((
      ('epochs', epochs), ('batch size', batch_size), ('items', train_loader_count, validate_loader_count),
      ('time', time.time() - total_time), ('accuracy', (vd_correct * 100 / validate_loader_count))
    ))

  def test(self, data_dir: str, labels: dict, batch_size: int = 64, freeze_parameters: bool = False):
    """
      Tests the model.

      Parameters
      ----------
        data_dir: str
          The path to the data directory.
        labels: dict
          The labels to test with.
        batch_size: int
          The batch size to use.
        freeze_parameters: bool
          Whether to freeze the parameters.

      Returns
      -------
        None
      """
    if 'test' not in self.transforms.keys():
      raise ValueError('Test or valid transforms must be defined and set')

    t_correct = 0
    ix = 0
    start_time = time.time()
    category_labels_ids = { v:k for k,v in enumerate(self.model.labels)}
    test_ldr = DataLoader(UnLabeledImageDataset(
      f"{data_dir}/test", labels, category_labels_ids, transform = self.transforms['valid']
      ), batch_size = batch_size, shuffle=True
    )

    test_loader_count = len(test_ldr.dataset)
    self._print_test_header(batch_size, test_loader_count)

    if freeze_parameters: # TODO: I think isn't needed with torch.no_grad(), investigate
      for param in self.model.parameters():
        param.requires_grad = False

    with torch.no_grad():
      self.model.eval()
      self._spin(step='test')

      for (images, (labels, _)) in iter(test_ldr):
        ix += batch_size * 1
        self._spin(step='test')

        output = self.model(images)
        self._spin(step='test')

        ps = torch.exp(output)
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        t_correct += equals.sum().item()
        self._spin(step='test')

        self._loading( ix, test_loader_count, step = 'test')
      else:
        self._print_t_step(start_time, t_correct, test_loader_count)

  def predict_image(self, image: str, **kwargs) -> tuple[np.array, list[str]]:
    """
      Calls the predict method, by transforming the given image path to a Pil.Image

      Parameters
      ----------
        image: str
          The path to the image to predict.
        topk: int
          The number of predictions to return.

      Returns
      -------
        : tuple[np.array, list[str]]
          The top-k probabilities and the top-k class labels.

    """
    return self.predict(Image.open(image), **kwargs)

  def predict(self, image:ImageDT, topk:int = 5) -> tuple[np.array, list[str]]:
    """
      Predicts the class of an given image.

      Parameters
      ----------
        image: Pil.Image
          The image to predict.
        topk: int
          The number of predictions to return.

      Returns
      -------
        : tuple[np.array, list[str]]
          The top-k probabilities and the top-k class labels.
    """
    with torch.no_grad():
      self.model.eval()
      tensor_img = self.transforms['test'](image)
      logps = self.model(tensor_img.unsqueeze(0)) # from 3d to 4d [ introducing a batch dimension ]
      ps = logps[0] # Return to 3D [ no batches again ]
      ps_val, ps_idx = ps.topk(topk)

      label_predictions = [self.model.labels[x] for x in ps_idx.numpy()]
      probabilities = [ 100 * (1 / (1 + math.exp(-x))) for x in ps_val]

      return probabilities, label_predictions
