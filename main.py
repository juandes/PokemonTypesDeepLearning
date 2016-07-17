import read_images
import show_images

def main():
    dataset = read_images.load_images()
    show_images.first_image_per_type(dataset)


if __name__ == "__main__":
    main()