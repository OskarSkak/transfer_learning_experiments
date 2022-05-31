import sys
from matplotlib import pyplot

def summarize_diagnostics(epochs, loss, val_loss, accuracy, val_accuracy, name):
	# plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.plot(epochs, loss, color='blue', label='train')
    pyplot.plot(epochs, val_loss, color='orange', label='validation')
    pyplot.legend(loc="lower right")
    # plot accuracy
    pyplot.subplot(212)
    pyplot.xlabel('epoch')
    pyplot.ylabel('accuracy')
    pyplot.title('Classification Accuracy')
    pyplot.plot(epochs, accuracy, color='blue', label='train')
    pyplot.plot(epochs, val_accuracy, color='orange', label='validation')
    pyplot.legend(loc="lower right")

    # save plot to file
    pyplot.tight_layout()
    pyplot.savefig(f'./plots/{name}_plot.png')
    # pyplot.close()
    # pyplot.show()
