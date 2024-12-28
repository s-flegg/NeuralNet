class Data:
    image_bytes_read = 16  # get rid of header
    label_bytes_read = 8 # get rid of header
    # data
    training_images = []
    training_labels = []
    test_images = []
    test_labels = []

def next_img(purpose):
    f = "file placeholder"
    
    # images
    if purpose == "test":
        f = open("MNIST/t10k-images.idx3-ubyte", "rb")
    elif purpose == "train":
        f = open("MNIST/train-images.idx3-ubyte", "rb")
    else:
        print("invalid purpose")

    f.read(Data.image_bytes_read).hex()  # get rid of already read
    array = []
    for i in range(28):
        for j in range(28):
            array.append(
                int(f.read(1).hex(), 16)  # read one byte, convert to hex, then int
            )
    f.close()
    
    # labels
    if purpose == "test":
        f = open("MNIST/t10k-labels.idx1-ubyte", "rb")
    elif purpose == "train":
        f = open("MNIST/train-labels.idx1-ubyte", "rb")
    else:
        print("invalid purpose")

    f.read(Data.label_bytes_read).hex()  # get rid of already read
    label = int(f.read(1).hex(), 16)  # read one byte, convert to hex, then int
    f.close()

    Data.image_bytes_read += 28 * 28
    Data.label_bytes_read += 1
    return array, label


if __name__ == "__main__":

    import pygame

    pygame.init()
    screen = pygame.display.set_mode((28 * 25, 28 * 25))
    font = pygame.font.SysFont("Arial", 20)

    def next_pygame_img():
        colours, label = next_img('train')
        for i in range(28):
            for j in range(28):
                colour = colours[i * 28 + j]
                surf = pygame.Surface((25, 25))
                surf.fill((colour, colour, colour))
                screen.blit(surf, (j * 25, i * 25))
        screen.blit(
            font.render(str(label), True, (255, 255, 255)),
            (0, 0)
        )

    next_pygame_img()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                next_pygame_img()

        pygame.display.flip()
