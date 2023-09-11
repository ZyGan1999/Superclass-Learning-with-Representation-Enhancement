from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from wilds.common.grouper import CombinatorialGrouper
import hyperparameters as HP

def get_data():
    dataset = get_dataset(dataset="iwildcam", download=True)

    # ['region', 'year', 'y', 'from_source_domain']
    grouper = CombinatorialGrouper(dataset, ['location'])

    train_data = dataset.get_subset("train",
        transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    ),
    )
    print(len(train_data))
    train_loader = get_train_loader("standard", train_data, batch_size=HP.batch_size)


    test_data = dataset.get_subset("test",
    transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    ),
    )
    print(len(test_data))
    test_loader = get_train_loader("standard", test_data, batch_size=HP.batch_size)

    return train_loader, test_loader, grouper

#train_loader, test_loader, grouper = get_data()
#print(len(train_loader))
#print(len(test_loader))
#for x,y,metadata in train_loader:
#    z = grouper.metadata_to_group(metadata)
#    print(z)