import alsaaudio

audio = alsaaudio.Mixer()
vol = audio.getvolume()
vol = int(vol[0])
slide = -20
newVol = vol + slide
audio.setvolume(newVol)