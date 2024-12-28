class Data:
    bytes_read = 16 # get rid of header

def next_img():
    f= open("MNIST/train-images.idx3-ubyte", "rb")
    f.read(Data.bytes_read).hex() # get rid of already read
    array = []
    for i in range(28):
        for j in range(28):
            array.append(
                int(f.read(1).hex(), 16) # read one byte, convert to hex, then int
            )
    f.close()
    Data.bytes_read += 28 * 28
    return array

if __name__ == "__main__":
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((28 * 25, 28 * 25))
    def next_pygame_img():
        colours = next_img()
        for i in range(28):
            for j in range(28):
                colour = colours[i * 28 + j]
                surf = pygame.Surface((25, 25))
                surf.fill((colour, colour, colour))
                screen.blit(surf, (j * 25, i * 25))

    next_pygame_img()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                next_pygame_img()

        pygame.display.flip()


