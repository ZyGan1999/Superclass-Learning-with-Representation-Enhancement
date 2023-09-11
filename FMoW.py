from cgi import test
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from wilds.common.grouper import CombinatorialGrouper
import hyperparameters as HP

'''
# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="fmow", download=True)

grouper = CombinatorialGrouper(dataset, ['region'])
# Get the training set
train_data = dataset.get_subset(
    "train",
    transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    ),
)

# Prepare the standard data loader
train_loader = get_train_loader("standard", train_data, batch_size=16)

for x,y,metadata in train_loader:
    z = grouper.metadata_to_group(metadata)
    print(z)

'''
def get_train_data():
    '''
    return a train loader and a grouper
    the grouper can transfer metadata to region
    '''
    dataset = get_dataset(dataset="fmow", download=True)

    # ['region', 'year', 'y', 'from_source_domain']
    grouper = CombinatorialGrouper(dataset, ['region'])

    train_data = dataset.get_subset("train",
    transform=transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()]
    ),
    )
    train_loader = get_train_loader("standard", train_data, batch_size=HP.batch_size)

    return train_loader, grouper

def get_test_data():
    dataset = get_dataset(dataset="fmow", download=True)

    # ['region', 'year', 'y', 'from_source_domain']
    grouper = CombinatorialGrouper(dataset, ['region'])

    test_data = dataset.get_subset("test",
    transform=transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()]
    ),
    )
    test_loader = get_train_loader("standard", test_data, batch_size=HP.batch_size)

    return test_loader, grouper

def get_data():
    dataset = get_dataset(dataset="fmow", download=True)

    # ['region', 'year', 'y', 'from_source_domain']
    grouper = CombinatorialGrouper(dataset, ['region'])

    train_data = dataset.get_subset("train",
        transform=transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()]
    ),
    )
    train_loader = get_train_loader("standard", train_data, batch_size=HP.batch_size)


    test_data = dataset.get_subset("test",
    transform=transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()]
    ),
    )
    test_loader = get_train_loader("standard", test_data, batch_size=HP.batch_size)

    return train_loader, test_loader, grouper


def get_sample_test_data():
    dataset = get_dataset(dataset="fmow", download=True)

    # ['region', 'year', 'y', 'from_source_domain']
    grouper = CombinatorialGrouper(dataset, ['region'])

    train_data = dataset.get_subset("test",
        transform=transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()]
    ),
    )

'''
# Prepare the standard data loader
train_loader = get_train_loader("standard", train_data, batch_size=16)

# (Optional) Load unlabeled data
dataset = get_dataset(dataset="iwildcam", download=True, unlabeled=True)
unlabeled_data = dataset.get_subset(
    "test_unlabeled",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)
unlabeled_loader = get_train_loader("standard", unlabeled_data, batch_size=16)

# Train loop
for labeled_batch, unlabeled_batch in zip(train_loader, unlabeled_loader):
    x, y, metadata = labeled_batch
    unlabeled_x, unlabeled_metadata = unlabeled_batch
    print(x,y)
'''