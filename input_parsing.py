import time


class Data:
    """
    Contains data for use in the file

    Attributes:
        training_images (list): contains the images for the training set stored as rows of greyscale values, 28x28
        training_labels (list): contains the labels for the training set
        test_images (list): contains the images for the test set stored as rows of greyscale values, 28x28
        test_labels (list): contains the labels for the test set
    """
    image_bytes_read = 16  # get rid of header
    label_bytes_read = 8  # get rid of header

    # data
    training_images = []
    training_labels = []
    test_images = []
    test_labels = []


def next_img(read_image_file, read_label_file):
    # images
    array = []
    for i in range(28):
        for j in range(28):
            array.append(
                int(read_image_file[Data.image_bytes_read:Data.image_bytes_read + 1].hex(),
                    16
                )/225  # read one byte, convert to hex, then int, then reduce range to 0-1
            )
            Data.image_bytes_read += 1


    # labels
    label_num = int(
        read_label_file[Data.label_bytes_read:Data.label_bytes_read + 1].hex(),
        16
    )  # read one byte, convert to hex, then int
    Data.label_bytes_read += 1
    label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    label[label_num] = 1

    return array, label

start_time = time.time()

# create array of labels and images
f_image = open("MNIST/train-images.idx3-ubyte", "rb")
f_label = open("MNIST/train-labels.idx1-ubyte", "rb")
b_image = f_image.read()
b_label = f_label.read()
f_image.close()
f_label.close()
for i in range(60000):
    colours, label = next_img(b_image, b_label)
    Data.training_images.append(colours)
    Data.training_labels.append(label)


Data.image_bytes_read = 16  # get rid of header
Data.label_bytes_read = 8 # get rid of header
f_image = open("MNIST/t10k-images.idx3-ubyte", "rb")
f_label = open("MNIST/t10k-labels.idx1-ubyte", "rb")
b_image = f_image.read()
b_label = f_label.read()
f_image.close()
f_label.close()
for i in range(10000):
    colours, label = next_img(b_image, b_label)
    Data.test_images.append(colours)
    Data.test_labels.append(label)

print("Parsing images from file took %f seconds" % (time.time() - start_time))

if __name__ == "__main__":

    start_time = time.time()
    ftri = open("MNIST/train.images.txt", "w")
    ftei = open("MNIST/test.images.txt", "w")
    ftrl = open("MNIST/train.labels.txt", "w")
    ftel = open("MNIST/test.labels.txt", "w")
    for i in range(60000):
        if i % 1000 == 0:
            print("Writing image %i of 60000" % i)
        ftri.write(str(Data.training_images[i]) + "\n")
        ftrl.write(str(Data.training_labels[i]) + "\n")
    for i in range(10000):
        if i % 1000 == 0:
            print("Writing image %i of 10000" % i)
        ftei.write(str(Data.test_images[i]) + "\n")
        ftel.write(str(Data.test_labels[i]) + "\n")
    ftri.close()
    ftei.close()
    ftrl.close()
    ftel.close()
    print("Writing took %f seconds" % (time.time() - start_time))

    # mainly for fun
    def test():
        import pygame

        pygame.init()
        screen = pygame.display.set_mode((28 * 25, 28 * 25))
        font = pygame.font.SysFont("Arial", 20)

        def next_pygame_img():
            colours, label = next_img("train")
            for i in range(28):
                for j in range(28):
                    colour = colours[i * 28 + j]
                    surf = pygame.Surface((25, 25))
                    surf.fill((colour, colour, colour))
                    screen.blit(surf, (j * 25, i * 25))
            screen.blit(font.render(str(label), True, (255, 255, 255)), (0, 0))

        next_pygame_img()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    next_pygame_img()

            pygame.display.flip()
