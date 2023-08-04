import matplotlib.pyplot as plt

# Save the bar chart of all train data
def save_bar(df):
    plt.title('seed_classification numbers')
    plt.bar(df.Catagory.unique(),
        df.Catagory.value_counts(), 
        width=0.8, 
        align='center', 
    )
    plt.xticks(rotation='vertical')
    for classes, num in zip(df.Catagory.unique(), df.Catagory.value_counts()):
        plt.text(classes, num, num, ha='center', va='bottom')
    plt.savefig('output/seed_classification.png')

# Save training accuracy plot
def save_acc(train_acc, valid_acc):
    plt.figure('Acc')
    plt.title('Accuracy')
    plt.plot(train_acc, linestyle='-', color='g')
    plt.plot(valid_acc, linestyle='-', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend(['train_accuracy','validatation_accuracy' ])
    plt.savefig('output/acc.png')

# Save training loss plot
def save_loss(train_loss, valid_loss):
    plt.figure('Loss')
    plt.title('Loss')
    plt.plot(train_loss, linestyle='-', color='g')
    plt.plot(valid_loss, linestyle='-', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train_loss','validatation_loss' ])
    plt.savefig('output/loss.png')