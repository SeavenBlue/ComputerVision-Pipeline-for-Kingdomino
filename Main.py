import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
import os

logs = False

# we need to look at terrain and crowns.
def main_loop():
    # Open tkinter window for one image or whole image folder to process:
    image_paths, automatic = choose_input_menu()
    if not image_paths:
        print("No images selected or found.")
        return

    results = [] # [file name, points] for every board

    # Loop through all the images and process them one by one
    for file_path in image_paths:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Issue loading {os.path.basename(file_path)}, skipping.")
            continue

        print("""
            ╭───────────────────────────────────────────────────────────────────────────────────────────────────────────╮
            │                                   ─═•⊰⊹  """,(f"Image ({os.path.basename(file_path)}) loaded..."),"""  ⊹⊱•═─                                │
            ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────╯   
        """)
        
        # HueSaturationvalue = show_hsv_channels(image)
        if not automatic: # when running automatically we dont open any windows
            cv2.imshow("Original: ", image)

        terrainArray = get_terrain_image(image,show = not automatic)
        CrownArray = get_crowns_image(image,show = not automatic)


        # count total points for the board
        points = count_points(terrainArray,CrownArray)
        print(f"Board {os.path.basename(file_path)}: {points} points")
        results.append([os.path.basename(file_path),points])

        #print_matrix("Tile Type: ",terrainArray)
        #print_matrix("Crown Matrix: ",CrownArray)

        # manually: wait for a key press before going on to the next image
        if not automatic:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # when running automatically, the list of points for every board gets printed here
    if automatic:
        print("Points per board:")
        for i in range(len(results)):
            print(f"Board {results[i][0]} = {results[i][1]} points")





"""
 ╭────────────────────────────────────────────────────────────────────────────╮
 │     ─═•⊰⊹ FUNCTIONS BELOW IN USE FOR TERRAIN TYPE CATEGORIZATION ⊹⊱•═─     │
 ╰────────────────────────────────────────────────────────────────────────────╯
"""

def get_terrain_image(image,show = True):
    global logs

    # Chose below what type of tile to focus on regarding terminal usage, multiple can be chosen
    #        0        1         2           3        4         5           6      7       8
    opt = ["Lakes","Fields","Grasslands","Forests","Mines","Wastelands","Castle",None,"Progress"]
    termprint = [opt[7]]


    termprint = np.array(termprint,dtype = str)



    # Lets get the overall brightness first
    overallbrightness = get_brightness_from_HSV(convert_array_to_HSV([image])[0])
    #print("Average Brightness: ",overallbrightness)
    
    # then we seperate the whole image into tiles so we can find out what each tile is seperately.
    tileArray = Seperator(image)
    tileSize = tileArray[0].shape[0]
    # Converting tiles to HSV as this format is less affected by light
    HSVTiles = convert_array_to_HSV(tileArray)

    # lets create the matrix which will hold our tiles once found. 
    tileMatrix = [[None,None,None,None,None],[None,None,None,None,None],[None,None,None,None,None],[None,None,None,None,None],[None,None,None,None,None]]


    """
    ╭────────────────────────────────╮
    │     ─═•⊰⊹ L A K E S ⊹⊱•═─     │
    ╰────────────────────────────────╯
    """    
    if (np.any(termprint == "Lakes")):
            print(" ─═•⊰⊹ L A K E S ⊹⊱•═─ ")
            logs = True
    else: logs = False   


    # To detect a lake is quite easy, lets remove light and then do a mean to find blue tiles
    
    # Removing brightness from image, by setting it to always 200
    fixedBrightnessTiles = use_effect_on_array(HSVTiles,set_all_pixels,None,None,200)
    #display_HSV_tiles(fixedBrightnessTiles,"Fixed Brightness Tiles")

    # now we trim off the edges of each tile and cut out the middle
    # This is so buildings and edge overlaps won't affect the tile color
    edgeArrayLakes = use_effect_on_array(convert_array_to_BGR(fixedBrightnessTiles),cutout,60,"mean",10) # uses mean to fill out the cuts
    
    # then we take a mean to get the average color of each tile
    meanArray = use_effect_on_array(edgeArrayLakes,get_mean)

    # with this we can try and detect lakes:
    lakeArray = use_effect_on_array(convert_array_to_HSV(meanArray),HSV_lake_detector)

    HSVMeanArray = convert_array_to_HSV(meanArray)
    log("Running: HSV_lake_detector")
    for i in range(len(lakeArray)):
        log("> Tile: ",i+1," Detected as: ",lakeArray[i]," with color: ", HSVMeanArray[i],"(HSV)")

    # and now we put our detections into our matrix:
    tileMatrix = array_onto_matrix(lakeArray,tileMatrix)

    #display_matrix(tileMatrix,tileSize,"Temp")


    '''
    ╭──────────────────────────────────╮
    │     ─═•⊰⊹ F I E L D S ⊹⊱•═─     │
    ╰──────────────────────────────────╯
    '''
    if(np.all(termprint == "Fields")):
            print(" ─═•⊰⊹ F I E L D S ⊹⊱•═─ ")
            logs = True
    else: logs = False

    # with lakes done, lets go the next tile type: Fields!
    # why? because they're still easy... ish

    # Lets start with a blur to remove shadows and other artifacts
    # blurredTiles = use_effect_on_array((tileArray),GaussBlur)
    
    # We'll now try to get the mode of all tiles (mode is the highest occuring color)
    # Note that mode uses bins, to identify differnt pixels which are basically the same color 
    modeArray = use_effect_on_array(tileArray,get_mode,20)
    #visualize_array_of_colors_BGR(modeArray,tileSize,"Temp")

    # with mode, we still want to get a mean of the tiles as well, Mode can help ignore houses and crowns
    # but mean is better at getting the actual color, so we can try and take the mean from the tile where a mill wouldn't be
    edgeArrayFields = use_effect_on_array(tileArray,cutout,60,"mean",10) # uses mean to fill out the cuts
    meanArray = use_effect_on_array(edgeArrayFields,get_mean)
    
    # now that we have the mode and the mean we can combine them, using weights. 
    # We use a slight weight to prefer mean over mode, this is because the mean ignores buildings already so the mode only cathes slight details the mean might overlook
    combinedMeanMode = (np.array(meanArray)*0.7 + np.array(modeArray)*0.3).astype(int) # Note this is weighted
    # visualize_array_of_colors_BGR(combinedMeanMode,tileSize,"Tempsy")
    #now we detect fields:
    
    fieldArray = diffirentiator_above_mean(convert_array_to_HSV(combinedMeanMode),0.9,HSV_field_detector,overallbrightness)

    # and now we put our detections into our matrix: 21
    tileMatrix = array_onto_matrix(fieldArray,tileMatrix)
    #display_matrix(tileMatrix,tileSize,"Testing")


    '''
    ╭──────────────────────────────────────────╮
    │     ─═•⊰⊹ G R A S S L A N D S ⊹⊱•═─     │
    ╰──────────────────────────────────────────╯
    '''

    # so, time for the last of the easy ones, grasslands are bright and green so we can generally detect them from that,
    # the major problem is distinguishing them from forrests.
    if(np.any(termprint == "Grasslands")):
        print(" ─═•⊰⊹ G R A S S L A N D S ⊹⊱•═─ ")
        logs = True
    else: logs = False

    # First, we will remove the middle of all tiles, this is to prevent houses to have a major effect on the output
    edgeArray = edgeArrayFields
    #visualize_array_of_colors_BGR(edgeArray,tileSize,"CutOut mean visualized:")
    # we will now get the mean for the whole image again, just to get it out as an array.
    meanArray = meanArray
    #visualize_array_of_colors_BGR(meanArray,tileSize,"Mean visualized:")

    edgeArray2 = use_effect_on_array(tileArray,cutout,40,"mode",12,20)
    #visualize_array_of_colors_BGR(edgeArray2,tileSize,"Cutout mode visualized(Grasslands):")
    modeArray = use_effect_on_array(edgeArray2,get_mode,20)
    #visualize_array_of_colors_BGR(modeArray,tileSize,"Mode visualized(Grasslands)")
    combinedMeanMode = (np.array(meanArray)*0.7 + np.array(modeArray)*0.3).astype(int) # Note this is weighted
    #visualize_array_of_colors_BGR(combinedMeanMode,tileSize,"Combined visualized(Grasslands):")

    # alright, we just have a tiny issue now, some of the images are very bright while others are quite dark
    # this means that we can't just use V(brightness) to see the difference
    # so we need to find the overall brightness of the image and then check if the tile is brighter than that
    
    # with that we can detect grass lands!
    grasslandArray = diffirentiator_above_mean(convert_array_to_HSV(combinedMeanMode),0.8,HSV_grassland_detector,overallbrightness)

    #temp = use_effect_on_array(convert_array_to_HSV(combinedMeanMode),HSV_grassland_detector,overallbrightness)
    #tileMatrix = array_onto_matrix(grasslandArray,tileMatrix)
    #display_matrix(tileMatrix,tileSize,"Testing:")

    # since forests are really similar to some grasslands we'll have to an additional check, we can use a threshold and then an erosion and dilution to remove incorrect detections
    # we can then use this additional information to detect whether there could be a forest or not
    confirmedGrasslandArray = confirm_grassland(grasslandArray,tileArray,[0,40],11,1000)
    
    # and now we put our detections into our matrix once again:
    tileMatrix = array_onto_matrix(confirmedGrasslandArray,tileMatrix)


    #gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    # now we use the threshold to find pixels within the threshold
    #binary = cv2.inRange(gray, 0, 40)
    #kernel = np.ones([11,11],dtype=np.uint8)
    #eroded = cv2.erode(binary,kernel)
    #diluted = cv2.dilate(eroded,kernel)

    #cv2.imshow("binary Image",diluted)
    #display_matrix(tileMatrix,tileSize,"Testing")
    
    #handle_images(image)

    '''
    ╭────────────────────────────────────╮
    │     ─═•⊰⊹ F O R E S T S ⊹⊱•═─     │
    ╰────────────────────────────────────╯
    '''
    
    if(np.any(termprint == "Forests")):
        print("─═•⊰⊹ F O R E S T S ⊹⊱•═─ ")
        logs = True
    else: logs = False

    # The time has come, to find the darker version of grasslands, simply put we do the same but now we want darker tiles.
    # First, we will remove the middle of all tiles, this is to prevent houses to have a major effect on the output
    edgeArray = use_effect_on_array(tileArray,cutout,50,"mean",15)
    
    # we will now get the mean for the whole image again, just to get it out as an array.
    meanArray = use_effect_on_array(edgeArray,get_mean)
    #visualize_array_of_colors_BGR(meanArray,tileSize,"Visualzied forest mean")
    
    #edgeArray2 = use_effect_on_array(tileArray,cutout,40,"mode",12,20)
    #visualize_array_of_colors_BGR(edgeArray2,tileSize,"Cutout mode visualized:")
    modeArray = use_effect_on_array(tileArray,get_mode,30)
    #visualize_array_of_colors_BGR(modeArray,tileSize,"Cutout mode visualized:")
    
    combinedMeanMode = (np.array(meanArray)*0.8 + np.array(modeArray)*0.2).astype(int) # Note this is weighted    
    #visualize_array_of_colors_BGR(combinedMeanMode,tileSize,"combinedefgag visualized:")

    log("Forest detections:")
    forestArray = []
    HSVMeanMode = convert_array_to_HSV(combinedMeanMode)
    for i in range(len(combinedMeanMode)):
        log("> Tile: ",i+1," has color: ",HSVMeanMode[i])
        
        forestArray.append(HSV_forest_detector(HSVMeanMode[i],overallbrightness,tileArray[i]))

    log("")

    

    # and then we simply confirm with our forest detector
    confirmedForestArray = confirm_forest(forestArray,tileArray,[0,45],11,20,confirmedGrasslandArray)

    #tileMatrix = array_onto_matrix(forestArray,tileMatrix,overwrite = True)
    #display_matrix(tileMatrix,tileSize,"Tesstyyyyyy")

    tileMatrix = array_onto_matrix(confirmedForestArray,tileMatrix,overwrite = False)

   
    #display_matrix(tileMatrix,tileSize,"ForestDone")

    


    '''
    ╭────────────────────────────────────────╮
    │         ─═•⊰⊹ M I N E S ⊹⊱•═─         │
    ╰────────────────────────────────────────╯
    '''
    if(np.any(termprint == "Mines")):
            print("─═•⊰⊹ M I N E S ⊹⊱•═─ ")
            logs = True
    else: logs = False
    # to detect mines we can check for the yellow middle which they all have and then the dark outside. 
    # we can reuse the mean and mode combined array to look for the dark surroundings.

    combinedMeanMode = (np.array(meanArray)*0.2 + np.array(modeArray)*0.8).astype(int) # Note this is weighted
    
    mineArray = confirm_mines(tileArray,combinedMeanMode,overallbrightness)

    tileMatrix = array_onto_matrix(mineArray,tileMatrix,overwrite = False)
    #display_matrix(tileMatrix,tileSize,"minesadded")
    #temp = use_effect_on_array(tileArray,cutout_keep_mid,10)
    #display_BGR_tiles(temp,"mineshafts")
    '''
    ╭──────────────────────────────────────────────────╮
    │         ─═•⊰⊹ W A S T E L A N D S ⊹⊱•═─         │
    ╰──────────────────────────────────────────────────╯
    '''
    if(np.any(termprint == "Wastelands")):
            print("─═•⊰⊹ W A S T E L A N D S ⊹⊱•═─ ")
            logs = True
    else: logs = False
    meanArray = use_effect_on_array(tileArray,get_mean)
    visualize_array_of_colors_BGR(meanArray,tileSize,"Mean visualized:")
        
    modeArray = use_effect_on_array(tileArray,get_mode,40)
    visualize_array_of_colors_BGR(modeArray,tileSize,"Mode visualized:")
        
    combinedMeanMode = (np.array(meanArray)*0.6 + np.array(modeArray)*0.4).astype(int) # Note this is weighted    
    #visualize_array_of_colors_BGR(combinedMeanMode,tileSize,"Combined visualized:")


    wastelandArray = []
    log("Running: detect_wastelands:")
    log("> Average brightness of image: ",overallbrightness)
    HSVarray = convert_array_to_HSV(combinedMeanMode)
    for i in range(len(tileArray)):
        log("> Tile: ",i+1, ". Color: ",HSVarray[i],"(HSV)")
        wastelandArray.append(HSV_detect_wastelands(HSVarray[i],overallbrightness))
    log("")

   
    removedEdge = use_effect_on_array(tileArray,cutout, 0,"mean", 20)
    #display_BGR_tiles(removedEdge,"Removed Edge: ")

    confirmedWastelandArray = confirm_wastes(wastelandArray,removedEdge)
    tileMatrix = array_onto_matrix(confirmedWastelandArray,tileMatrix,overwrite = False)


    '''
        ╭──────────────────────────────────────────╮
        │         ─═•⊰⊹ C A S T L E ⊹⊱•═─         │
        ╰──────────────────────────────────────────╯
    '''
    # The culmination of tile detection, the one tile with the highest amount of difference.
    # A castle tile can either be a flat surface or there can be a 3d castle on top which makes it very difficult to detect.
    # without the castle the base tile has differnt colors as well, but we can try and look for the crown in the middle of the tile
    if(np.any(termprint == "Castle")):
        print("─═•⊰⊹ C A S T L E ⊹⊱•═─ ")
        logs = True
    else: logs = False

    
    # if there is an uneven amount of tiles still not detected then we know the castle is one of them
    completedTileArray = matrix_to_array(tileMatrix)
    log("Running: Castle Logic:")
    log("Current detections: ",completedTileArray)
    undetectedAmount = 0
    for i in range(len(completedTileArray)):
        if completedTileArray[i] is None:
            undetectedAmount = undetectedAmount+1
    validCastleTiles = np.zeros(len(completedTileArray))
    if undetectedAmount % 2 == 0:
        log("> Castle is not part of the undetected tiles")
        for i in range(len(completedTileArray)):
            if completedTileArray[i] is None:
                validCastleTiles[i] = -100

    else:
        log("> Castle is one of the undetected tiles")
        for i in range(len(completedTileArray)):
            if completedTileArray[i] is not None:
                validCastleTiles[i] = -100

    log("")
    log("> Remaining valid locations:")
    for i in range(len(validCastleTiles)):
        if validCastleTiles[i] >= 0:
            log("- > Tile: ",i+1," Remains a valid option.")

    log("")

    
    visualize_array_of_colors_BGR(combinedMeanMode,tileSize,"testing")
    blurredTiles = use_effect_on_array(tileArray,cv2.GaussianBlur,[5,5],0)
    edgeDetection = use_effect_on_array(blurredTiles,cv2.Canny,0,200)

    display_BGR_tiles(edgeDetection,"CrownTile Detector:")

    template = load_single_template("CrownTile.png")
    
    log("Running: template_matching:")
    for i in range(len(tileArray)):
        validCastleTiles[i] = validCastleTiles[i] + template_matching(tileArray[i],template)
        if validCastleTiles[i] > 0:
            log("> Tile: ",i+1," New castle tile score: ",validCastleTiles[i])



    # Now we'll just keep building on the percentages
    # Crown tiles seem to hae one color, and then they're mostly gray.
    # Lets check grayscale then
    
    grayScale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    grayScaleBGR = cv2.cvtColor(grayScale,cv2.COLOR_GRAY2BGR)
    cv2.imshow("Grayscale: ",grayScaleBGR)
    grayMean = np.array(use_effect_on_array(Seperator(grayScaleBGR),get_mean),np.uint8)
    
    generalMean = np.array(use_effect_on_array(tileArray,get_mean),int)
    
    # we'll subtrack grayscale from the BGR and find the difference, high differnce equals lower score
    for i in range(len(validCastleTiles)):
        diff = np.sum([abs(generalMean[i][0]-grayMean[i][0]),abs(generalMean[i][1]-grayMean[i][1]),abs(generalMean[i][2]-grayMean[i][2])],None,float)
        log("> Tile: ",i+1," has: ",diff," in difference from grayscale")
        points = 1-diff/100.
        validCastleTiles[i] = validCastleTiles[i]+points
        if validCastleTiles[i] > 0:
            log("- > Contender for Castle with: ",validCastleTiles[i],"score")
    

    # We could do more, but since we've already confirmed a lot of the tiles and ensure they don't overlap the castle tile, we can simply say that the castle can't be on top of those
    detectionArray = matrix_to_array(tileMatrix)
    log(detectionArray)
    for i in range(len(validCastleTiles)):
        if  detectionArray[i] == "Lake":
            validCastleTiles[i] = validCastleTiles[i]-100
        elif  detectionArray[i] == "Field":
            validCastleTiles[i] = validCastleTiles[i]-100
        elif  detectionArray[i] == "Grassland":
            validCastleTiles[i] = validCastleTiles[i]-100
        elif  detectionArray[i] == "Mine":
            validCastleTiles[i] = validCastleTiles[i]-0.1


    

    pos = np.argmax(validCastleTiles)
    castleResultArray = np.full(len(validCastleTiles), None, dtype=object)
    castleResultArray[pos] = "Castle"

    tileMatrix = array_onto_matrix(castleResultArray,tileMatrix,True)
    

    display_matrix(tileMatrix,tileSize,"Tiles detected so far:")
    
    return tileMatrix








"""
 ╭─────────────────────────────────────────────╮
 │     ─═•⊰⊹ FUNCTIONS BELOW IN USE ⊹⊱•═─     │
 ╰─────────────────────────────────────────────╯
"""

def crown_tile_by_contour(tile):

    tileGray = cv2.cvtColor(tile,cv2.COLOR_BGR2GRAY)
    
    blurTile = cv2.GaussianBlur(tileGray, (7, 7), 0)
    thresholdedTile = cv2.threshold(blurTile, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # 4. Find contours from the thresholded binary image
    contours = cv2.findContours(thresholdedTile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        return None

    # 5. Find the largest contour (this will be our central crown/castle)
    largest_cnt = max(contours, key=cv2.contourArea)
    
    # 6. Measure Area and Arc Length (Perimeter)
    area = cv2.contourArea(largest_cnt)
    perimeter = cv2.arcLength(largest_cnt, closed=True)
    
    print(f"Crown Contour -> Area: {area:.1f}, Perimeter (Arc Length): {perimeter:.1f}")
    
    # 7. Define your validation thresholds 
    # (Adjust these numbers based on your tile resolution, e.g., if tiles are ~100x100 pixels)
    min_area = 200      # Too small = just a speck of noise
    min_perimeter = 60  # Too short = incomplete shape
    max_area = 3500     # Too large = covers the whole tile
    crownResult = False
    if area > min_area and area < max_area:
        if perimeter > min_perimeter:
            crownResult = True
  
    return [crownResult, area, perimeter]


def load_single_template(filename):
    # Loads a single template image
    

    file_path = os.path.join(templateFolder, filename)
    
    template = cv2.imread(file_path)
    if template is None:
        raise FileNotFoundError(f"Could not find {filename} in {templateFolder}")
        
    return template


def template_matching(Tile,Template):

    tileGray = cv2.cvtColor(Tile,cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(Template,cv2.COLOR_BGR2GRAY)

    width = tileGray.shape[1]
    height = tileGray.shape[0]

    # ensuring tile and template is same size
    templateGray = cv2.resize(templateGray, (width, height), interpolation=cv2.INTER_AREA)

    resultArray = []
    for i in range(4): # running 4 times for each 90 degree template rotations
        templateGray = cv2.rotate(templateGray,cv2.ROTATE_90_CLOCKWISE)

        result = cv2.matchTemplate(tileGray, templateGray, cv2.TM_CCOEFF_NORMED)[0][0]
        if result < 0: result = 0
        else: result = round(result,3) 
        resultArray.append(result)

    output = max(resultArray)
    return output


def confirm_wastes(wasteArray, trimmedTiles, minValDiff = 75):
    log("Running: confirm_wastes:")
    confirmed = []
    for i in range(len(wasteArray)):
        if wasteArray[i] == "Wasteland":
            edgeCaseBright = find_edge_cases_among_pixels(trimmedTiles[i])[2]
            log("> Tile: ",i+1,"Low val pixel: ",edgeCaseBright[0], ", High val pixel: ", edgeCaseBright[1])
            if edgeCaseBright[1] < 100: minValDiff = minValDiff- 25; log("- > Low val in tile, lowering diff threshold")
            if (edgeCaseBright[1]-edgeCaseBright[0] > minValDiff):
                log("- > Tile: ",i+1," Passed val difference check")
                confirmed.append("Wasteland")
            else:
                confirmed.append(None)
                log("- > Tile: ",i+1," Failed val difference check, minimum diff: ",minValDiff)
        else:
            confirmed.append(None)


    log("")
    return confirmed

def find_edge_cases_among_pixels(Tile):

    tile = np.array(Tile.copy(), dtype = np.uint8)

    width = tile.shape[1]
    height = tile.shape[0]
    
    val1 = []
    val2 = []
    val3 = []
    
    for y in range(height):
        for x in range(width):
            val1.append(tile[y,x][0])
            val2.append(tile[y,x][1])
            val3.append(tile[y,x][2])
    
    high1 = max(val1)
    high2 = max(val2)
    high3 = max(val3)
    low1 = min(val1)
    low2 = min(val2)
    low3 = min(val3)
    
    return [[low1,high1],[low2,high2],[low3,high3]]







def HSV_detect_wastelands(color,avgBrightness):
    # accepts only a single pixel color
    wasteHue = [10,30]
    wasteSat = [20,205]
    wasteVal = np.array([avgBrightness*0.5,avgBrightness*1.1],dtype = np.uint8)
    tileHue = color[0]
    tileSat = color[1]
    tileVal = color[2]
    result = None
    if (tileHue >= wasteHue[0] and tileHue <= wasteHue[1]):
        if(tileSat >= wasteSat[0] and tileSat <= wasteSat[1]):
            if(tileVal >= wasteVal[0] and tileVal <= wasteVal[1]):

                result = "Wasteland"
                log("- > Detected as Wasteland")
            else: log("- > Value out of range: ",wasteVal)
        else: log("- > Saturation out of range: ",wasteSat)
    else: log("- > Hue out of range: ",wasteHue)

    return result


def log(*toPrint):
    if logs:
        print(*toPrint)
    return logs

def find_pixels_based_on_thresholds(Tile,lower,upper):
    # can use both HSV and BGR
    tile = np.array(Tile.copy(), dtype = np.uint8)
    width = tile.shape[1]
    height = tile.shape[0]
    detections = 0
    averageOfDetections = []
    for y in range(height):
        for x in range(width):
            if np.all(tile[y,x] > lower) and np.all(tile[y,x] < upper):
                detections = detections +1
                averageOfDetections.append(tile[y,x])

    if detections == 0:
        return detections , np.array([0,0,0], dtype = np.uint8)

    averageOfDetections = np.array(averageOfDetections,dtype= float)
    averageColor = [np.sum(averageOfDetections[:,0])/detections,np.sum(averageOfDetections[:,1])/detections,np.sum(averageOfDetections[:,2])/detections]
    averageColor = np.array(averageColor,dtype = np.uint8)

    return detections , averageColor






def confirm_mines(tile_array,meanModeArray,avgbright,allowance = 10, modeBin = 10):
    log("Running: confirm_mines:")
    mineDetectorOutput = []
    HSVMeanMode = convert_array_to_HSV(meanModeArray)
    for i in range(len(HSVMeanMode)):
        mineDetectorOutput.append(HSV_mine_detector(HSVMeanMode[i],avgbright))
        log("> Tile: ",i+1, ". HSV: ",HSVMeanMode[i], ". Detected as: ",mineDetectorOutput[i][0])
    
    

    donutMode = [item[0] for item in mineDetectorOutput]
    donutModeColor = [item[1] for item in mineDetectorOutput]


    donutHoles = use_effect_on_array(tile_array,cutout_keep_mid,modeBin)
    holeMode = np.array(use_effect_on_array(donutHoles,get_mode,modeBin))
    confirmed = []
    averageBGR = np.array([0,0,0],dtype = np.float64)
    runningAmount = float(0)

    log("")

    lowerMine = [0, 70, 90]
    upperMine = [65, 130, 150]
    minAmount = 500

    for i in range(len(tile_array)):
        if donutMode[i] == "Mine":
            tileMean = (holeMode[i][0].astype(np.float64)+holeMode[i][1].astype(np.float64)+holeMode[i][2].astype(np.float64))//3
            tileMean = tileMean.astype(np.uint8)
            log("> Confirming tile: ",i+1,", Mineshaft BGR: ",holeMode[i],". average: ",tileMean)
            if (holeMode[i][0] < tileMean) and (holeMode[i][2] > tileMean) and (holeMode[i][2] > 60):
                log("- > Tile Passed as mine")
                confirmed.append("Mine")
                averageBGR = averageBGR+convert_array_to_BGR([donutModeColor[i]])[0]
                runningAmount = runningAmount+1
            else: 
                
                amount , avgcolor = find_pixels_based_on_thresholds(tile_array[i],lowerMine,upperMine)
                log("- > Tile declined, Using amount check instead. Amount: ", amount,". with average color: ",avgcolor)
                if amount > minAmount: 
                    log("- - > Accepted as mine")
                    confirmed.append("Mine")
                else:
                    
                    log("- - > Declined, Attempting combined amount and color palette spread: ")
                    if (amount > minAmount/2) and (holeMode[i][0] < tileMean) and (holeMode[i][2] > tileMean):
                        log("- - - > Accepted, based on: amount: ",amount," > ",minAmount/2," and [B: ",holeMode[i][0]," < ",tileMean," ,R: ",holeMode[i][2]," > ",tileMean)
                        confirmed.append("Mine")
                    else:
                        confirmed.append(None)
                        log("- - - > Declined, Not a Mine")

                
            

        else:
            confirmed.append(None)
    
    # some forest with huts in blurry images still detect as mines, so we can make a secondary check, where we compare the average BGR values of the donut
    log("")
    if(runningAmount > 0):
        averageBGR = np.array([averageBGR[0]//runningAmount,averageBGR[1]//runningAmount,averageBGR[2]//runningAmount],dtype = np.uint8)
        log("> Average ground BGR: ", averageBGR, ". Upper limit: ", [averageBGR[0]+allowance,averageBGR[1]+allowance,averageBGR[2]+allowance])
        for i in range(len(tile_array)):
            if confirmed[i] == "Mine":
                log("> Tile: ",i+1,": validating confirmation...")
                if  (donutModeColor[i][0] < averageBGR[0]+allowance) and (donutModeColor[i][1] < averageBGR[1]+allowance) and (donutModeColor[i][2] < averageBGR[2]+allowance):
                    log("- > Confirmed tile as mine")
                else:
                    confirmed[i] = None
                    log("- > Not validated as a mine")

    return confirmed


def cutout_keep_mid(tile,modeBin):
    #def cutout(tile,percentage,fill = "mode", trim = 10, *args):
    # This function removes the middle of a tile and replaces it with either the mode or the mean of the rest
    # Percentage defines what amount from the middle is cut out where 100% is the whole tile

    tileWidth = tile.shape[1] 
    tileHeight = tile.shape[0]

    result = tile.copy()

    pct = 0.25
    
    centerX = tileWidth//2
    centerY = tileHeight//2

    cutSizeX = int(centerX*pct)
    cutSizeY = int(centerY*pct)     

    centerMask = np.zeros([tileHeight,tileWidth],dtype=bool)
    centerMask[centerY-cutSizeY:centerY+cutSizeY,centerX-cutSizeX:centerX+cutSizeX] = True
    

    center = tile[centerMask]
    centerColor = get_mode(center,modeBin)


    result[~centerMask] = centerColor
  
    return result.astype(np.uint8)




def HSV_mine_detector(color,avgbright):
    # accepts only a single pixel color
    # we should be able detect mines by finding lower saturations.
    mineHue = [12,35]
    mineVal = [0,avgbright/2]
    mineBGR = 50
    
    tileHue = color[0]
    tileVal = color[2]
    tileBGR = convert_array_to_BGR([color])[0]
    result = None
    if (tileHue >= mineHue[0] and tileHue <= mineHue[1]):
        if(tileVal >= mineVal[0] and tileVal <= mineVal[1]):
            if (tileBGR[0] < mineBGR) and (tileBGR[1] < mineBGR) and (tileBGR[2] < mineBGR):

              
                result = "Mine"
       
    return [result, tileBGR.astype(np.uint8)]


def remove_shadows_bgr(image, blur_kernel=(151, 151)):
    # 1. Load the image in BGR
    img = image

    # 2. Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 3. Split channels and cast the V channel to float32 for math accuracy
    h, s, v = cv2.split(hsv)
    v_float = v.astype(np.float32)

    # 4. Create the blurred background illumination layer
    # A large kernel is essential to capture overall ambient lighting rather than local textures
    v_blurred = cv2.GaussianBlur(v_float, blur_kernel, 0)

    # 5. Divide original V by the blurred background
    # We use a scale factor of 255 to map the result back to the 0-255 grayscale range
    v_corrected_float = cv2.divide(v_float, v_blurred, scale=255)

    # 6. Clip values to [0, 255] boundary limits and cast back to standard 8-bit integer
    v_corrected = np.clip(v_corrected_float, 0, 255).astype(np.uint8)

    # 7. Merge the corrected V channel back with original untouched H and S channels
    final_hsv = cv2.merge([h, s, v_corrected])

    # 8. Convert back to standard BGR for display or saving
    result = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return result





def confirm_forest(forestArray,originalArray,thresh,erosion,target,grassArray):
    log("Running: confirm_forest:")
    log("> Minimum accepted amount of forest: ",target)
    confirmedArray = []
    runningVal = 0
    runs = 0
    camps = np.array([[0,20,40],[10,45,65]],dtype = np.uint8)
    meanArray = use_effect_on_array(use_effect_on_array(originalArray,cutout,50,"mean",15),get_mean)
    HSVMeanArray = convert_array_to_HSV(meanArray)


    # Check if a confirmed tile is already detected as grass, then we remove it from the forest array
    for i in range(len(forestArray)):
        if grassArray[i] == "Grassland" and forestArray[i] == "Forest":
            forestArray[i] = None
            log("> Detected forest placement on top of grassland, tile: ",i+1 ,". Removing it before proceeding")


    for i in range(len(forestArray)):
        log("> Tile: ",i+1,", detected as: ",forestArray[i])
        confirmed = None
        if forestArray[i] == "Forest":
            if (forest_based_on_blobs(originalArray[i],thresh,erosion,target)):
                runs = runs+1
                runningVal = runningVal + int(HSVMeanArray[i][2])    
                confirmed = "Check"
                log("- > Confirmed as: ",forestArray[i])
            else:
                campCheck = np.array(get_mean(cutout_keep_mid(originalArray[i],20)),dtype = np.uint8)
                log("- > Looking for camp: ",campCheck )

                if np.all(campCheck >= camps[0]) and np.all(campCheck <= camps[1]):
                    log("- > Confirmed camp, accepting tile as: ",forestArray[i])
                    runs = runs+1
                    runningVal = runningVal + int(HSVMeanArray[i][2])        
                    confirmed = "Check"

                else:
                    log("- > No camp found.")

        confirmedArray.append(confirmed)          

    log(" ")
    if runs == 0:
        log("> No forests found on this board")
        return confirmedArray

    
    averageVal = runningVal//runs
    log("> Average Brightness(Val): ",averageVal, ". Min: ",max(averageVal*0.6-20,0), ". Max: ",averageVal*1.5+20)
    for i in range(len(confirmedArray)):
        if confirmedArray[i] == "Check":
            log("> Tile: ",i+1," has: ", HSVMeanArray[i][2])
            if (HSVMeanArray[i][2] >= max(averageVal*0.6-20,0)) and (HSVMeanArray[i][2] <= (averageVal*1.5+20)):
                confirmedArray[i] = "Forest"
                log("- > Confirmed by Brightness(Val)")
            else:
                log("- > Brightness(Val) doesn't match with other forests")
                confirmedArray[i] = None
        
    return confirmedArray



def HSV_forest_detector(color,avgBrightness,tile):
    # accepts only a single pixel color
    forestHue = [28,90]
    forestSat = [40,255]
    forestVal = np.array([0,avgBrightness*1.2],dtype = np.uint8)
    tileHue = color[0]
    tileSat = color[1]
    tileVal = color[2]
    result = None
    if (tileHue >= forestHue[0] and tileHue <= forestHue[1]):
        if(tileSat >= forestSat[0] and tileSat <= forestSat[1]):
            if(tileVal >= forestVal[0] and tileVal <= forestVal[1]):

                result = "Forest"
            else: log("- > Value out of range: ",forestVal)
        else: log("- > Saturation out of range: ",forestSat)
    else: log("- > Hue out of range: ",forestHue)

    
    if result != "Forest":
        camp = np.array([[0,2,20],[10,45,65]],dtype = np.uint8)
        campCheck = np.array(get_mean(cutout_keep_mid(tile,20)),dtype = np.uint8)
        log("- - > Checking for camp, color: ",campCheck,". camp range: ",camp[0]," to ",camp[1])
        if np.all(campCheck >= camp[0]) and np.all(campCheck <= camp[1]):
            newHue = [forestHue[0]-3,forestHue[1]-30]
            log("- - > Camp detected, new accepted hue range: ",newHue)
            if (tileHue >= newHue[0] and tileHue <= newHue[1]):
                result = "Forest"
                log("- - > Accepted as forest.")
            else: log("- - > Declined as forest.")
        else: log("- - > No camp detected")
    
    return result


def confirm_grassland(grassArray,originalArray,thresh,erosion,target):
    log("Running: confirm_grassland:")
    log("> Maximum accepted amount of forest: ",target)
    confirmedArray = []
    for i in range(len(grassArray)):
        log("> Tile: ",i+1,", detected as: ",grassArray[i])
        confirmed = None
        if grassArray[i] == "Grassland":
            if not (forest_based_on_blobs(originalArray[i],thresh,erosion,target)):
                confirmed = "Grassland"
                log("- > Confirmed as: ",grassArray[i])
                
            
        confirmedArray.append(confirmed)

    return confirmedArray
    
    




def forest_based_on_blobs(image,thresh,erosion,target):
    couldBeForest = False
    blobs = detect_blobs(image,thresh,erosion)
    blobamount = 0
    for y in range(blobs.shape[0]):
        for x in range(blobs.shape[1]):
            if blobs[y,x] == 255:
                blobamount = blobamount+1

    log("- > Amount of Forest in tile: ",blobamount)
    if blobamount > target:
        couldBeForest = True

    return couldBeForest
    
    

def detect_blobs(image,thresh,erosion):
    # First we convert the image to grayscale so we can try and create a threshold
    image = image.astype(np.uint8)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Gray Image",gray)
    # now we use the threshold to find pixels within the threshold
    binaryTile = cv2.inRange(gray, thresh[0], thresh[1])

    if erosion % 2 == 0: #ensures the kernel is an uneven number
        erosion = erosion-1

    # then we apply erosion to remove wrong detections
    kernel = np.ones([erosion,erosion],dtype=np.uint8)
    eroded = cv2.erode(binaryTile,kernel)
    diluted = cv2.dilate(eroded,kernel)


    return diluted

def cutout(tile,percentage,fill = "mode", trim = 10, *args):
    # This function removes the middle of a tile and replaces it with either the mode or the mean of the rest
    # Percentage defines what amount from the middle is cut out where 100% is the whole tile

    tileWidth = tile.shape[1] 
    tileHeight = tile.shape[0]

    result = tile.copy()

    pct = percentage / 100.0 if percentage > 1.0 else percentage

    centerX = int(tileWidth//2)
    centerY = int(tileHeight//2)

    cutSizeX = int(centerX*pct)
    cutSizeY = int(centerY*pct)

    edgeMask = np.ones([tileHeight,tileWidth],dtype=bool)
    edgeMask[centerY-cutSizeY:centerY+cutSizeY,centerX-cutSizeX:centerX+cutSizeX] = False
    edgeMask[0:trim,0:tileWidth] = False
    edgeMask[tileHeight-trim:tileHeight,0:tileWidth] = False
    edgeMask[0:tileHeight,0:trim] = False
    edgeMask[0:tileHeight,tileWidth-trim:tileWidth] = False
    

    edge = tile[edgeMask]
    
    if fill is "mode":
        mid = get_mode(edge,*args)
    elif fill is "mean":
        runningamount = 0
        runningValue = np.zeros(3,dtype = float)
        for i in range(len(edge)):
            runningamount = runningamount+1
            runningValue = runningValue + edge[i]
        mid = (runningValue//runningamount).astype(np.uint8)
    else:
        log("Wrong fill function name. please check spelling")
        return

    result[centerY-cutSizeY:centerY+cutSizeY,centerX-cutSizeX:centerX+cutSizeX] = mid
    result[0:trim,0:tileWidth] = mid
    result[tileHeight-trim:tileHeight,0:tileWidth] = mid
    result[0:tileHeight,0:trim] = mid
    result[0:tileHeight,tileWidth-trim:tileWidth] = mid

    return result
  



def diffirentiator_above_mean(array, allowance, function, *args):
    log("Running: diffirentiator_above_mean:")
    #Insert a function to use like: HSV_field_detector
    #This function checks if values are below a mean to remove those values
    # It then also checks if the hue value is below a certain range based off detections, to remove wrong types of tiles.

    allowance = np.array(allowance,dtype=float) # given amount that tiles can vary

    runningAverage = np.array([0,0,0]) 
    runningAmount = 0
    detectedArray = []
    tempResult = []
    hueMin = []

    # for loop below finds the average hue, sat and val of all tiles which could be the correct type
    for i in range(len(array)):
        log("> tile: ",i+1,". Color: ",array[i])
        tempResult = function(array[i],*args)
        if tempResult is not None:
            log("-> detected as: ",tempResult)
            runningAmount =  runningAmount+1
            detectedArray.append(array[i])
            runningAverage = runningAverage + np.array(array[i])

    if runningAmount is not 0:
        meandetected = ((runningAverage//runningAmount)*allowance).astype(int)
    else:
        log("- - Found none of specified tile type... - -")
        empty = np.zeros(len(array),dtype = int)
        return empty
    log(" ")
    log("> mean found: ",meandetected," with minimum values: ",meandetected*0.8)
    # now we use the average of the possibly correct tiles, to figure out which tiles are above average
    # we then use those abvove average tiles to determine a precise minimum hue of the tiles, which can help remove the wrong tiles

    for i in range(len(array)):
        tempResult = function(array[i],*args)
        if tempResult is not None:
            if array[i][1] >= meandetected[1] and array[i][2] >= meandetected[2]*0.9:
                hueMin.append(array[i][0])
    
    hueMax = max(hueMin)
    hueMin = min(hueMin)
    log(" ")
    log("> Hue minimum found: ",hueMin,". Hue maximum: ",hueMax)
    log("- > allowance modified min: ",hueMin*allowance,". max: ",2*hueMax-hueMax*allowance)
    # with this, we can now make an overall filtering using our new findings.
    trueDetects = []
    for i in range(len(array)): 
        truth = False
        tempResult = function(array[i],*args)
        if tempResult is not None:
            if array[i][1] >= meandetected[1]*0.9 and array[i][2] >= meandetected[2]*0.8:
                if (array[i][0] >= hueMin*allowance and array[i][0] <= (2*hueMax-hueMax*allowance)):
                    log("> tile: ",i+1,". color: ",array[i],". is confirmed based on filters.")
                    truth = True
        if truth:
            trueDetects.append(tempResult)
        else:
            trueDetects.append(None)

    log("- - - - - - - - - -")
    log(" ")
    return trueDetects


def get_brightness_from_HSV(image):
    # image sizes
    imageWidth = image.shape[1] 
    imageHeight = image.shape[0]
    runningSum = 0
    runningAmount = 0
    
    for y in range(imageHeight):
        for x in range(imageWidth):
            runningAmount = runningAmount+1
            runningSum = runningSum + image[y,x,2].astype(int)

    
    HSVMean = runningSum//runningAmount
    return HSVMean


def get_mode(image,binSize = 20):
    
    pixelArray = []  

    if image.ndim == 3:
        # image sizes
        imageWidth = image.shape[1] 
        imageHeight = image.shape[0]
          
        for y in range(imageHeight):
            for x in range(imageWidth):
                pixelArray.append(image[y,x])

    else:
        pixelArray = image
            
    # now that we have collected all pixels, we need to sort them into colors
    # we can do this by dividing each pixel by an amount to reduce their respective difference

    # first we make the differences in color less noticable so we can find the most common color
    simillarArray =[]
    for i in range(len(pixelArray)):
        simillarArray.append(pixelArray[i]//binSize)

    # now we can then go ahead and find the most often occuring color
    [colorbin,votes] = np.unique(simillarArray, return_counts=True, axis=0)
    # we now have how many there are of each of the colors, so now we just need to find the position of the bin with most votes.
    color = ((colorbin[np.argmax(votes)])*binSize).astype(np.uint8)
    #log("mode for given image (rgb): ",color[2],color[1],color[0])

    return color





def GaussBlur(image):
    # Here we want to repetedly blur the image until we have just 25 pixels left,
    # This will be beneficial when identifying terrain
    tempIMG = image
    #creating the kernel to blur the image with

    #setting the height and width of the image:
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]

    # now we must find the height of each section so we can pixelize the image
    pixelHeight = imageHeight//4
    pixelWidth = imageWidth//4
    #pixel sizes must be odd, so we have to add 1 to the size if they are even
    if (pixelHeight % 2 == 0):
        pixelHeight += 1
    if (pixelWidth % 2 == 0):
        pixelWidth += 1

    #Now we apply a gausian blur on the image
    tempIMG = cv2.GaussianBlur(tempIMG, (pixelWidth, pixelHeight),0)
    

    return tempIMG



def HSV_grassland_detector(color,avgBrightness):
    # accepts only a single pixel color
    grasslandHue = [32,52]
    grasslandSat = [130,255]
    grasslandVal = [min(110,pow(avgBrightness,2)/200) ,255]
    tileHue = color[0]
    tileSat = color[1]
    tileVal = color[2]
    result = None
    if (tileHue >= grasslandHue[0] and tileHue <= grasslandHue[1]):
        if(tileSat >= grasslandSat[0] and tileSat <= grasslandSat[1]):
            if(tileVal >= grasslandVal[0] and tileVal <= grasslandVal[1]):

                
                result = "Grassland"
   
    return result


def HSV_field_detector(color,avgBrightness):
    # accepts only a single pixel color
    fieldHue = [20,27]
    fieldSat = [210,255]
    fieldVal = [min(110,avgBrightness*0.9),255]
    tileHue = color[0]
    tileSat = color[1]
    tileVal = color[2]
    result = None
    
    
    if (tileHue >= fieldHue[0] and tileHue <= fieldHue[1]):
        if(tileSat >= fieldSat[0] and tileSat <= fieldSat[1]):
            if(tileVal >= fieldVal[0] and tileVal <= fieldVal[1]):

                result = "Field"
       
    return result

def handle_images(image,finish = False):
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if not finish:
        cv2.imshow("Original: ",image)

def matrix_to_array(matrix):
    #We assume an n*n matrix
    matrix = np.array(matrix)
    newArray = matrix.copy().flatten().tolist()
    return newArray

def display_matrix(matrix,size,name):
    #first convert it into an array
    array = matrix_to_array(matrix)
    visualArray = np.zeros((len(array),3),dtype=np.uint8)

    #now we use defined colors for each tile (BGR)
    uncategorized = [0,0,0]
    lake = [250,100,0]
    field = [50,200,250]
    grassland = [100,250,100]
    forest = [0,100,0]
    mine = [50,50,50]
    wasteland = [60,120,120]
    castle = [150,255,255]


    for i in range(len(array)):
        if array[i] == 0 or array[i] is None:
            visualArray[i] = uncategorized
        elif array[i] == "Lake":
            visualArray[i] = lake
        elif array[i] == "Field":
            visualArray[i] = field
        elif array[i] == "Grassland":
            visualArray[i] = grassland
        elif array[i] == "Forest":
            visualArray[i] = forest
        elif array[i] == "Mine":
            visualArray[i] = mine
        elif array[i] == "Wasteland":
            visualArray[i] = wasteland
        elif array[i] == "Castle":
            visualArray[i] = castle

    
    visualize_array_of_colors_BGR(visualArray,size,name)

def array_onto_matrix(array,matrix,overwrite = True):
    #before we begin we must check that the array can fill the matrix
    #also the matrix must be n*n 
    if(len(array) == pow(len(matrix),2)):
        conSize = int(np.floor(np.sqrt(len(array))))
        newMatrix = matrix.copy()
        
        for i in range(len(array)):
            x = int(i % conSize)
            y = int(np.floor(i/conSize))
            if type(array[i]) == str:
                if overwrite:
                    newMatrix[y][x] = array[i]  
                elif newMatrix[y][x] is None:
                    newMatrix[y][x] = array[i]

        return newMatrix

    else:
        log("given array and matrix do not match in size, array length: ",len(array), " Matrix size: ",pow(len(matrix),2) )

def HSV_lake_detector(color):
    # accepts only a single pixel color
    lakeHue = [90,110]
    tileHue = color[0]
    lakeSat = [65,255]
    tileSat = color[1]
    result = None
    
    if (tileHue >= lakeHue[0] and tileHue <= lakeHue[1]):
        if(tileSat >= lakeSat[0] and tileSat <= lakeSat[1]):
            result = "Lake"
       
    return result

def Seperator(image):
    # This function is to separate the image into 25 sections, so we can identify the terrain of each section
    
    imageWidth = image.shape[1] 
    imageHeight = image.shape[0]
    groupSizeX = imageWidth // 5 #divides the image up into 5 equal bits to find each tile
    groupSizeY = imageHeight // 5

    groupedSections = []
    for y in range(5):
        for x in range(5):
            section = image[y*groupSizeY:(y+1)*groupSizeY, x*groupSizeX:(x+1)*groupSizeX]
            groupedSections.append(section)
            

    return groupedSections

def visualize_array_of_colors_BGR(array,size,name):

    tileArray = []
            
    for i in range(len(array)):
                
        tile = np.zeros([size,size,3],dtype = np.uint8)
        tile[:][:] = array[i]
        tileArray.append(tile)
        
    display_BGR_tiles(tileArray,name)
    return None

def get_mean(image):
    # image sizes
    imageWidth = image.shape[1] 
    imageHeight = image.shape[0]
    runningSumR = 0
    runningSumG = 0
    runningSumB = 0
    runningAmount = 0

    for y in range(imageHeight):
        for x in range(imageWidth):
            runningAmount = runningAmount+1
            runningSumR = runningSumR + image[y,x,2].astype(int)
            runningSumG = runningSumG + image[y,x,1].astype(int)
            runningSumB = runningSumB + image[y,x,0].astype(int)

    tileMean = np.zeros([3],dtype=np.uint8)
    tileMean = ([runningSumB//runningAmount,runningSumG//runningAmount,runningSumR//runningAmount])

    return tileMean

def use_effect_on_array(array,effect, *args):
    #this function calls a given function on an array
    # *args will stash all extra arguments
    newArray = []

    for i in range(len(array)):
        newArray.append(effect(array[i],*args))

    return newArray

def set_all_pixels(image,value1 = None,value2 = None,value3 = None):
    # this function goes in to each pixel and sets a value to a specific amount 
    # if a value is None then that value wil not be changed in any pixel
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]

    newImage = image.copy() # makes it into an exact copy of the image.

    
    if value1 is not None:
        newImage[:, :, 0] = value1
    if value2 is not None:
        newImage[:, :, 1] = value2
    if value3 is not None:
        newImage[:, :, 2] = value3

    # we just changed all pixels values and we can now just return the new image
    return newImage

def modify_all_pixels(image,value1 = None,value2 = None,value3 = None):
    # this function goes in to each pixel and sets a value to a specific amount 
    # if a value is None then that value wil not be changed in any pixel
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]

    newImage = image.copy() # makes it into an exact copy of the image.

    
    if value1 is not None:
        newImage[:, :, 0] = newImage[:, :, 0]*value1
    if value2 is not None:
        newImage[:, :, 1] = newImage[:, :, 1]*value2
    if value3 is not None:
        newImage[:, :, 2] = newImage[:, :, 2]*value3

    # we just changed all pixels values and we can now just return the new image
    return newImage

def convert_array_to_HSV(tileArray):
    HSVArray = []
    for i in range(len(tileArray)):
        # converting to numpy array to check dimensions of array
        element = np.array(tileArray[i], dtype=np.uint8)
        
        # detect whether this is an image/tile
        if element.ndim == 3:
            HSVArray.append(cv2.cvtColor(element, cv2.COLOR_BGR2HSV))
            
        # if it wasnt an image/tile then it is a singular color value
        else:
            # we can force cv2 to change it to HSV if we make it look like a complete image
            packedColor = np.array([[element]], dtype=np.uint8)
            HSVColor = cv2.cvtColor(packedColor, cv2.COLOR_BGR2HSV)
            # now we unpack the color which is now in HSV, and append it
            HSVArray.append(HSVColor[0][0])    

    return HSVArray

def convert_array_to_BGR(tileArray):
    BGRArray = []
    for i in range(len(tileArray)):
        # converting to numpy array to check dimensions of array
        element = np.array(tileArray[i], dtype=np.uint8)
            
        # detect whether this is an image/tile
        if element.ndim == 3:
            BGRArray.append(cv2.cvtColor(element, cv2.COLOR_HSV2BGR))
                
        # if it wasnt an image/tile then it is a singular color value
        else:
            # we can force cv2 to change it to BGR if we make it look like a complete image
            packedColor = np.array([[element]], dtype=np.uint8)
            BGRColor = cv2.cvtColor(packedColor, cv2.COLOR_HSV2BGR)
            # now we unpack the color which is now in BGR, and append it
            BGRArray.append(BGRColor[0][0])
                
    return BGRArray

def display_HSV_tiles(tileArray,name):
    # This function can be used to display an array of HSV tiles, as if they were combined
    #     
    array = convert_array_to_BGR(tileArray)
    display_BGR_tiles(array,name)

    return None

def display_BGR_tiles(tileArray,name):
    #create first stack of tiles (rows) then stack those stacks in y and we have an image

    conSize = int(np.floor(np.sqrt(len(tileArray))))

    column = []
    
    for i in range(conSize):
        row = np.hstack(tileArray[i*conSize:(i+1)*conSize])
        column.append(row)

    #now we stacked it and we just need to display it
    tileImage = np.array(np.vstack(column),dtype=np.uint8)
    cv2.imshow(name,tileImage)

    return None


"""
 ╭────────────────────────────────────────────────────────────────────────────╮
 │     ─═•⊰⊹ FUNCTIONS ABOVE IN USE FOR TERRAIN TYPE CATEGORIZATION ⊹⊱•═─     │
 ╰────────────────────────────────────────────────────────────────────────────╯

 
 ###############    ANOTHER MAIN FUNCTION    ###############

 
 ╭────────────────────────────────────────────────────────────────╮
 │     ─═•⊰⊹ FUNCTIONS BELOW IN USE FOR CROWN DETECTION ⊹⊱•═─     │
 ╰────────────────────────────────────────────────────────────────╯
"""


######### Crown feature-space characteristics #########

# the gold in the middle of a crown (HSV), used to see if its golden in the middel
crownHSVLow  = (20,100,100)
crownHSVHigh = (33,255,255)

# template matching:
matchThreshold = 0.85       # Match thredshold
rescueThreshold = 0.76      # Lowest thredshold
templateScales = [1.0,0.85] # Crown size differenences
peakDistance = 9            # Two blobs or one blob?
minCentreGold = 0.3         # Is the center of blob golden?
border = 24                 # padding to find Crowns on the edge of image
templateFolder = os.path.dirname(os.path.abspath(__file__))

# smudge check, for crowns so blurred that the white outline is gone:
smudgeWindow = 24           # half size of the window we look at around the match
minBackgroundSat = 180      # the background has to be strongly coloured (field yellow, lake blue) for the smudge to stand out
satDrop = 90                # the smudge must be at least this much less saturated than the background
smudgeArea = [150,600]      # a blurred crown including its halo is roughly 20x15 to 27x22 pixels
maxSmudgeAspect = 1.8
minSmudgeFill = 0.45

# rim check, for crowns partly hidden behind a building or a neighbouring tile:
goldArea = [40,300]         # the gold inside of a single crown, in pixels
maxGoldSize = 26            # width and height of that gold patch
minRimContrast = 25         # the rim just outside the gold has to be this much whiter than the gold itself




def get_crowns_image(image,show = True):
    # first we find every crown on the whole board, then we sort them into the 25 tiles
    crowns = find_crowns(image)
    crownArray = crowns_onto_array(image,crowns)

    # lets show where we found them
    display_crowns(image,crowns)

    return crownArray



def find_crowns(image):
    # the template cant be centered on a crown that is cut off by the edge of the picture,
    # so we pad the board by repeating the edge pixels (and remember to subtract it again at the end)
    paddedImage = cv2.copyMakeBorder(image,border,border,border,border,cv2.BORDER_REPLICATE)
    scoreImage = get_crown_scores(paddedImage)

    # a crown is a peak in the score image, so a pixel that is the highest within peakDistance of itself.
    # we cheat a bit here: dilate puts the local maximum into every pixel, so a pixel equal to its dilation is a peak
    se = np.ones((2*peakDistance+1,2*peakDistance+1),np.uint8)
    localMax = cv2.dilate(scoreImage,se)
    peaks = np.argwhere((scoreImage >= rescueThreshold) & (scoreImage == localMax))

    # the inside of a crown is gold.
    HSVImage = cv2.cvtColor(paddedImage,cv2.COLOR_BGR2HSV)
    goldMask = cv2.inRange(HSVImage,crownHSVLow,crownHSVHigh)
    whiteness = cv2.GaussianBlur(get_whiteness(paddedImage),(5,5),0)
    # we also label all the gold blobs now, the rim check needs them
    [goldCount,goldLabels,goldStats,goldCentroids] = cv2.connectedComponentsWithStats(goldMask,connectivity=8)

    crowns = []
    for i in range(len(peaks)):
        y = int(peaks[i][0])
        x = int(peaks[i][1])
        score = float(scoreImage[y,x])

        # is there gold in the middle?
        centre = goldMask[y-4:y+4,x-4:x+4]
        goldFraction = np.count_nonzero(centre)/centre.size
        if (goldFraction < minCentreGold):
            continue

        # a weak match only counts if it is a blurred out crown, or a crown that is partly hidden
        if (score < matchThreshold):
            if (not blurred_crown_detector(HSVImage,x,y)):
                if (not hidden_crown_detector(goldLabels,goldStats,whiteness,x,y)):
                    continue

        # back to the coordinates of the real picture, and throw away peaks that landed in the padding
        x = x-border
        y = y-border
        if (x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]):
            continue

        print("Crown found at: ",x,y," score: ",round(score,2))
        crowns.append([x,y,score])

    return crowns


# feature-space matching
def get_crown_scores(image):
    # we match on the whiteness image, because the white outline is what makes a crown look like a crown
    whiteness = get_whiteness(image)
    whiteness = cv2.GaussianBlur(whiteness,(5,5),0)

    imageWidth = image.shape[1]
    imageHeight = image.shape[0]

    # every pixel gets the best score any of the templates got there, -1 means no template fit at all
    bestScores = np.full([imageHeight,imageWidth],-1.0,np.float32)
    for i in range(len(crownTemplates)):
        template = crownTemplates[i]
        templateWidth = template.shape[1]
        templateHeight = template.shape[0]

        scores = cv2.matchTemplate(whiteness,template,cv2.TM_CCOEFF_NORMED)

        # matchTemplate gives the score at the top left corner of the window, we want it at the centre of the crown
        centeredScores = np.full([imageHeight,imageWidth],-1.0,np.float32)
        centeredScores[templateHeight//2:templateHeight//2+scores.shape[0], templateWidth//2:templateWidth//2+scores.shape[1]] = scores
        bestScores = np.maximum(bestScores,centeredScores)

    return bestScores



def blurred_crown_detector(HSVImage,x,y):
    # a crown that is blurred this much has no white outline left, it is just a grey smudge.
    # but on a field or a lake that smudge is still the only crown sized patch with low saturation,
    # swamp texture, sheep, table edges and plain noise dont make a patch like that
    result = False

    saturation = HSVImage[y-smudgeWindow:y+smudgeWindow, x-smudgeWindow:x+smudgeWindow, 1]
    if (saturation.shape != (2*smudgeWindow,2*smudgeWindow)):
        return result   # too close to the edge of the padded picture

    # background saturation = the median of an 8 pixel wide frame along the edge of the window
    frame = np.concatenate([saturation[:8].ravel(), saturation[-8:].ravel(), saturation[8:-8,:8].ravel(), saturation[8:-8,-8:].ravel()])
    background = np.median(frame)
    if (background < minBackgroundSat):
        return result

    # now we find blobs that are clearly less saturated than the background
    smudgeMask = (saturation < background-satDrop).astype(np.uint8)
    [blobCount,labels,stats,centroids] = cv2.connectedComponentsWithStats(smudgeMask,connectivity=8)

    for i in range(1,blobCount):
        blobX = stats[i][0]
        blobY = stats[i][1]
        blobWidth = stats[i][2]
        blobHeight = stats[i][3]
        area = stats[i][4]

        # the smudge has to sit on the match, be crown sized, be compact, and not run out of the window
        distance = np.hypot(centroids[i][0]-smudgeWindow, centroids[i][1]-smudgeWindow)
        if (distance > 5):
            continue
        if (area < smudgeArea[0] or area > smudgeArea[1]):
            continue
        if (blobX == 0 or blobY == 0 or blobX+blobWidth == 2*smudgeWindow or blobY+blobHeight == 2*smudgeWindow):
            continue
        fill = area/(blobWidth*blobHeight)
        aspect = max(blobWidth,blobHeight)/min(blobWidth,blobHeight)
        if (fill < minSmudgeFill or aspect > maxSmudgeAspect):
            continue

        print("^ blurred crown ^")
        result = True
        break

    return result



def hidden_crown_detector(goldLabels,goldStats,whiteness,x,y):
    # a crown with part of its outline hidden (behind a building, or under the edge of the next tile) scores lower,
    # because part of the outline is missing. but the gold inside is still a small compact patch, and the outline
    # that is visible is still whiter than the gold. a boat sail or a roof is gold with a white edge too, but way bigger
    result = False

    # which gold blob covers most of the 8x8 middle of the match?
    centreLabels = goldLabels[y-4:y+4,x-4:x+4]
    centreLabels = centreLabels[centreLabels > 0]
    if (len(centreLabels) == 0):
        return result
    label = np.bincount(centreLabels).argmax()

    blobX = goldStats[label][0]
    blobY = goldStats[label][1]
    blobWidth = goldStats[label][2]
    blobHeight = goldStats[label][3]
    area = goldStats[label][4]

    # the gold patch must be the size of the inside of one crown
    if (area < goldArea[0] or area > goldArea[1] or blobWidth > maxGoldSize or blobHeight > maxGoldSize):
        return result

    # the rim is a 2 pixel band just outside the gold patch, and it has to be whiter than the gold
    top = max(0,blobY-2)
    bottom = min(goldLabels.shape[0],blobY+blobHeight+2)
    left = max(0,blobX-2)
    right = min(goldLabels.shape[1],blobX+blobWidth+2)
    gold = (goldLabels[top:bottom,left:right] == label).astype(np.uint8)
    rim = cv2.dilate(gold,np.ones((5,5),np.uint8)) - gold
    if (rim.sum() == 0):
        return result
    regionWhiteness = whiteness[top:bottom,left:right]
    rimContrast = regionWhiteness[rim > 0].mean() - regionWhiteness[gold > 0].mean()
    if (rimContrast >= minRimContrast):
        print("^ partly hidden crown ^")
        result = True

    return result



def crowns_onto_array(image,crowns):
    # same order as Seperator (left to right, top to bottom), so print_matrix can reshape it into the 5x5 board
    tileWidth = image.shape[1]//5
    tileHeight = image.shape[0]//5
    crownArray = [0]*25

    for i in range(len(crowns)):
        x = crowns[i][0]
        y = crowns[i][1]
        column = min(4,x//tileWidth)
        row = min(4,y//tileHeight)
        crownArray[row*5+column] += 1

    return crownArray



def get_whiteness(image):
    # point processing: a pixel only gets a high value if it is bright (high V) AND grey/white (low S).
    # so the white outline of a crown lights up, while yellow, blue, green and the black of the mines all stay dark
    HSVImage = cv2.cvtColor(image,cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = HSVImage[:,:,1]
    value = HSVImage[:,:,2]
    whiteness = (255-saturation)*value/255

    return whiteness.astype(np.uint8)



def load_crown_templates():
    # the templates are two crowns cut out of the boards (one sharp, one blurry), made with make_templates.py
    templates = []
    for fileName in ["crown_sharp.png","crown_blurry.png"]:
        template = cv2.imread(os.path.join(templateFolder,fileName))
        if (template is None):
            raise FileNotFoundError("Could not find "+fileName+" in "+templateFolder+", run make_templates.py first")

        # we match on whiteness, so the template has to be whiteness too, blurred the same way as the board
        whiteness = get_whiteness(template)
        whiteness = cv2.GaussianBlur(whiteness,(5,5),0)

        for i in range(len(templateScales)):
            scaled = whiteness
            if (templateScales[i] != 1.0):
                scaled = cv2.resize(whiteness,None,fx=templateScales[i],fy=templateScales[i],interpolation=cv2.INTER_AREA)

            # crowns rotate with the tile, so we need all 4 orientations
            templates.append(scaled)
            templates.append(cv2.rotate(scaled,cv2.ROTATE_90_CLOCKWISE))
            templates.append(cv2.rotate(scaled,cv2.ROTATE_180))
            templates.append(cv2.rotate(scaled,cv2.ROTATE_90_COUNTERCLOCKWISE))

    return templates

crownTemplates = load_crown_templates()



def display_crowns(image,crowns):
    # draws a box around every crown we found and shows it
    boxedImage = image.copy()
    halfSize = 14   # a crown size plus a bit of buffer

    for i in range(len(crowns)):
        x = crowns[i][0]
        y = crowns[i][1]
        cv2.rectangle(boxedImage,(x-halfSize,y-halfSize),(x+halfSize,y+halfSize),(255,0,255),1)

    cv2.imshow("Detected crowns:",boxedImage)

    return None

"""
 ╭────────────────────────────────────────────────────────────────╮
 │     ─═•⊰⊹ FUNCTIONS ABOVE IN USE FOR CROWN DETECTION ⊹⊱•═─     │
 ╰────────────────────────────────────────────────────────────────╯
"""


###############################################       Helper Function       ###############################################
def show_hsv_channels(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv_image)

    cv2.imshow("Hue", hue)
    cv2.imshow("Saturation", saturation)
    cv2.imshow("Value", value)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



###############################################       Tkinter window       ###############################################
def ask_two_options(question,option1,option2):
    # opens a small window with a question and two buttons, and returns the text of the button that was pressed
    choise = None

    # Tkinter window:
    root = tk.Tk()
    root.title("Kingdomino")
    root.geometry("600x300")
    root.resizable(False,False)
    root.attributes("-topmost",True) # always on top
    root.eval('tk::PlaceWindow . center')

    def pick(option):
        nonlocal choise
        choise = option
        root.destroy()

    # Question?
    label = ttk.Label(root, text=question)
    label.pack(pady=20)

    # The two options/(buttons):
    buttonRow = ttk.Frame(root)
    buttonRow.pack()
    ttk.Button(buttonRow, text=option1, command=lambda: pick(option1)).pack(side="left", padx=10)
    ttk.Label(buttonRow, text="or").pack(side="left")
    ttk.Button(buttonRow, text=option2, command=lambda: pick(option2)).pack(side="left", padx=10)

    root.mainloop() # keeps window open until button is pressed

    return choise # stays None if the window is closed with the X

def choose_input_menu():
    choise = ask_two_options("Would you like to process:","One image","Image folder")
    if choise is None:
        return [], False # Stopped because user closed window

    automatic = False # Pre-sets the global variabel
    if choise is "Image folder":
        runMode = ask_two_options("How should the images be run?","Manually","Automatically")
        if runMode is None:
            return [], False # Stopped because user closed window
        automatic = runMode is "Automatically"

    return get_user_images(choise), automatic


###############################################       Get Image(-s)       ###############################################
imageTypes = [".jpg", ".jpeg", ".jfif", ".pjpeg", ".pjp", ".png", ".apng", ".webp",]
# Set accepted filetypes (from https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Formats/Image_types)  


def get_user_images(choise):

    root = tk.Tk()
    root.withdraw() # Hides the small tk window while in file exploror

    image_paths = []
    # For getting single image
    if choise == "One image":
        file_path = filedialog.askopenfilename(
            title = "Select a Kingdomino image.",
            filetypes = [("Images"," ".join("*" + fileTypes for fileTypes in imageTypes)), ("All files", "*.*")]
        )
        if file_path: # Empty if window is closed
            image_paths.append(file_path)

    # For getting images from image folder
    elif choise == "Image folder":
        folder_path = filedialog.askdirectory(title = "Select the folder wwith Kingdomino images")
        if folder_path:
            i = 1
            while True:
                # Looks for i."imageTypes"
                found = None
                for fileTypes in imageTypes:
                    path = os.path.join(folder_path, f"{i}{fileTypes}")
                    if os.path.isfile(path):
                        found = path
                        break
                if found is None: # Found no more images
                    break
                image_paths.append(found)
                i += 1 # Goes for next image

    root.destroy()
    return image_paths


def print_matrix(title, values):
    print(f"{title}")
    grid = np.array(values).reshape(5, 5)
    print(grid)
    print()  # empty line for readability





###############################################       Get Point(-s)       ###############################################

def count_points(tileMatrix,crownArray):
    # Terrain types
    terrains = ["Lake", "Field", "Grassland", "Forest", "Wasteland", "Mine", "Castle"]

    # Makes a tile and crown matrix
    tiles = np.array(tileMatrix).reshape(5,5)
    crowns = np.array(crownArray).reshape(5,5)
    blobIds = np.zeros([5,5],dtype=int) # For setting which tile crowns belongs to

    currentId = 1
    totalPoints = 0
    for y in range(5):
        for x in range(5):
            if blobIds[y,x] == 0 and tiles[y,x] in terrains:
                [area,crownCount] = burn_baby_burn(tiles,crowns,blobIds,(y,x),currentId)
                points = area * crownCount
                print(f"{tiles[y,x]}: {area} tiles x {crownCount} crowns = {points} points")
                totalPoints += points
                currentId += 1
                
    return totalPoints


def burn_baby_burn(tiles,crowns,blobIds,start,currentId):
    # burns every tile connected to start that has the same terrain and labels them with currentId
    terrain = tiles[start]
    burnQueue = [start]
    blobIds[start] = currentId
    area = 0
    crownCount = 0

    while len(burnQueue) > 0:
        [y,x] = burnQueue.pop()
        area += 1
        crownCount += int(crowns[y,x])

        # add connections to the burn queue: 4-connected, because diagonal tiles dont connect in Kingdomino
        for [ny,nx] in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]:
            if 0 <= ny < 5 and 0 <= nx < 5 and blobIds[ny,nx] == 0 and tiles[ny,nx] == terrain:
                blobIds[ny,nx] = currentId # burned as soon as it is added, so it cant be added twice
                burnQueue.append((ny,nx))
    
    return area,crownCount


   
#Now that we have all the functions loaded we start the main loop:
main_loop()
