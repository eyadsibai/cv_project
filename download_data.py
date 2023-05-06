# %%
from SoccerNet.Downloader import SoccerNetDownloader
path = 'data'
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=path)

mySoccerNetDownloader.password = "s0cc3rn3t"
mySoccerNetDownloader.downloadGames(
    files=["1_224p.mkv", "2_224p.mkv"],
    split=["train", "valid", "test",
           "challenge"])  # download 224p Videos (require password from NDA)

mySoccerNetDownloader.downloadGames(files=["1_player_boundingbox_maskrcnn.json", "2_player_boundingbox_maskrcnn.json"], split=[
                                    "train", "valid", "test", "challenge"])  # download Player Bounding Boxes inferred with MaskRCNN
