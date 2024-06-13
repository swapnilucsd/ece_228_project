import torch
from utils import get_mean_iou, get_pixel_accuracy
from tqdm import tqdm

def train(
    epochs,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    patch=False,
):
    """
    Train a model with given parameters, managing both training and validation phases.

    Args:
        epochs (int): Total number of epochs to train for.
        model (torch.nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        scheduler: Learning rate scheduler.
        device (torch.device): Device to train the model on.
        patch (bool): Whether to use patched inputs (default is False).

    Returns:
        dict: A dictionary containing training and validation losses and metrics.
    """
    model.to(device)
    train_losses, val_losses, train_iou, val_iou, train_acc, val_acc = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss, train_iou_score, train_accuracy = 0.0, 0.0, 0.0

        for data in tqdm(train_loader):
            images, masks = data
            if patch:
                images = images.view(-1, *images.size()[2:])
                masks = masks.view(-1, *masks.size()[2:])

            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou_score += get_mean_iou(outputs, masks)
            train_accuracy += get_pixel_accuracy(outputs, masks)

        train_losses.append(train_loss / len(train_loader))
        train_iou.append(train_iou_score / len(train_loader))
        train_acc.append(train_accuracy / len(train_loader))

        val_loss, val_iou_score, val_accuracy = validate(
            model, val_loader, criterion, device, patch
        )
        val_losses.append(val_loss)
        val_iou.append(val_iou_score)
        val_acc.append(val_accuracy)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_{epoch+1}.pth")
        else:
            patience_counter += 1

        if patience_counter >= 6:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        print_epoch_stats(
            epoch,
            epochs,
            train_losses[-1],
            val_losses[-1],
            train_iou[-1],
            val_iou[-1],
            train_acc[-1],
            val_acc[-1],
        )

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_iou": train_iou,
        "val_iou": val_iou,
        "train_acc": train_acc,
        "val_acc": val_acc,
    }


def print_epoch_stats(
    epoch, epochs, train_loss, val_loss, train_iou, val_iou, train_acc, val_acc
):
    """Prints the statistics for the epoch."""
    print(
        f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, "
        f"Train IoU: {train_iou:.3f}, Val IoU: {val_iou:.3f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}"
    )


def validate(model, loader, criterion, device, patch):
    """Validation step to evaluate the model on the validation dataset."""
    model.eval()
    total_loss, total_iou, total_acc = 0.0, 0.0, 0.0
    with torch.no_grad():
        for data in tqdm(loader):
            images, masks = data
            if patch:
                images = images.view(-1, *images.size()[2:])
                masks = masks.view(-1, *masks.size()[2:])

            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_iou += get_mean_iou(outputs, masks)
            total_acc += get_pixel_accuracy(outputs, masks)

    return total_loss / len(loader), total_iou / len(loader), total_acc / len(loader)
