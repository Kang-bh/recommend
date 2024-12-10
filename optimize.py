from sklearn.model_selection import RandomizedSearchCV
from models import Stage1Model
import torch.optim as optim
from train import train_model
from evaluate import evaluate_model

param_space = {
    'hidden_dim': [128, 256, 512],
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'batch_size': [32, 64, 128],
    'epochs': [5, 10, 15]
}

def objective(params):
    model = Stage1Model(item_dim=768, text_dim=768, hidden_dim=params['hidden_dim'])
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    
    train_model(model, train_loader, optimizer, params['epochs'])
    evaluate_model(model, test_data, books_df)
    
    return model

random_search = RandomizedSearchCV(
    estimator=objective,
    param_distributions=param_space,
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
