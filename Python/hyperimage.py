from scipy.io import loadmat
from matplotlib import pyplot as plt, colors as pltcolors


def show_labeled_image(labeled_img, block_window=True):
    # Semantic Classes for Urban Scenes
    myColors = ["#ffffff",  # id: 0, name: undefined, (0,0,0)
                "#ff0000",  # id: 1, name: Road (Ground - Drivable), (255,0,0)
                "#ff8000",  # id: 2, name: Sidewalk (Ground - Drivable), (255,128,0)
                "#ff00ff",  # id: 3, name: Lane Markers (Ground - Drivable), (255,0,255)
                "#00ff00",  # id: 4, name: Grass (Ground - Drivable), (0,255,0)
                "#00ff80",  # id: 5, name: Vegetation (Not Grass - Not Drivable), (0,255,128)
                "#5500dc",  # id: 6, name: Panels/Signs/TraficLight (Static Obstacles), (85,0,220)
                "#55009d",  # id: 7, name: Building/Wall/Others (Static Obstacles, (85,0,157)
                "#ffff00",  # id: 8, name: Car/Truck/Train/Bus/Plane/Bicycle/Motocycle/etc., (255,255,0)
                "#ffaac0",  # id: 9, name: Adult,Children,Cyclist,Motocyclist,Animal - (Person - Moveable Obstacle), (255,170,192)
                "#0000ff"]  # id: 10, name: Sky, (0,0,255)
    plt.imshow(labeled_img, cmap=pltcolors.ListedColormap(myColors), vmin=0, vmax=10)
    plt.axis('off')
    plt.title("")
    plt.show(block=block_window)


class Hyperimage:
    def __init__(self, mat_path):
        # load the file
        mat_data = loadmat(mat_path)

        # ['data', 'wavelengths', 'id_Semantic Classes for Urban Scenes', 'stats', 'image', 'id_spectral_reflectances', 'label_spectral_reflectances', 'label_Semantic Classes for Urban Scenes']
        self.wavelengths = mat_data['wavelengths']
        self.hypercube = mat_data['data']
        self.label_semantic_classes_for_urban_scenes = mat_data['label_Semantic Classes for Urban Scenes']

    def get_hypercube(self):
        return self.hypercube

    def get_wavelengths(self):
        return self.wavelengths

    def get_label(self):
        return self.label_semantic_classes_for_urban_scenes

    def get_label_spectral_reflectances(self):
        return self.label_spectral_reflectances

    def get_mat_data(self):
        return self.mat_data

    def show_hyper_img(self, block_window=True):
        plt.imshow(self.hypercube[:, :, 10])
        plt.axis('off')
        plt.title("")
        plt.show(block=block_window)

    def show_labeled_img(self, block_window=True):
        # Semantic Classes for Urban Scenes
        myColors = ["#ffffff",  # id: 0, name: undefined, (0,0,0)
                    "#ff0000",  # id: 1, name: Road (Ground - Drivable), (255,0,0)
                    "#ff8000",  # id: 2, name: Sidewalk (Ground - Drivable), (255,128,0)
                    "#ff00ff",  # id: 3, name: Lane Markers (Ground - Drivable), (255,0,255)
                    "#00ff00",  # id: 4, name: Grass (Ground - Drivable), (0,255,0)
                    "#00ff80",  # id: 5, name: Vegetation (Not Grass - Not Drivable), (0,255,128)
                    "#5500dc",  # id: 6, name: Panels/Signs/TraficLight (Static Obstacles), (85,0,220)
                    "#55009d",  # id: 7, name: Building/Wall/Others (Static Obstacles, (85,0,157)
                    "#ffff00",  # id: 8, name: Car/Truck/Train/Bus/Plane/Bicycle/Motocycle/etc., (255,255,0)
                    "#ffaac0",
                    # id: 9, name: Adult,Children,Cyclist,Motocyclist,Animal - (Person - Moveable Obstacle), (255,170,192)
                    "#0000ff"]  # id: 10, name: Sky, (0,0,255)
        plt.imshow(self.label_semantic_classes_for_urban_scenes, cmap=pltcolors.ListedColormap(myColors),
                   vmin=0, vmax=10)
        plt.axis('off')
        plt.title("")
        plt.show(block=block_window)