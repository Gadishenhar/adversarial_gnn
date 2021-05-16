from twitter_dataset import TweeterDataset
import pickle
import torch

if __name__ == '__main__':
    dataset = TweeterDataset('data/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('glove.pkl', 'rb') as file:
        glove_matrix = pickle.load(file)    # (vocab_size, dim) == (10000, 200)

    glove_matrix = torch.tensor(glove_matrix, dtype=torch.float32).to(device) # (200, 10000)

    initial_node_representations = torch.matmul(dataset[0].x, glove_matrix)

    print('initial_node_representations size:', initial_node_representations.size())
    print('Verify that there exist some words in every example:')
    print((dataset[0].x > 0).sum(dim=1))

