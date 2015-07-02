//
//  PlaySoundsViewController.swift
//  Pitch Perfect
//
//  Created by SiqiWu on 25/06/2015.
//  Copyright (c) 2015 Siqi. All rights reserved.
//

import UIKit
import AVFoundation

class PlaySoundsViewController: UIViewController {
    
    var audioPlayer: AVAudioPlayer!
    var audioEngine : AVAudioEngine!
    var audioFile: AVAudioFile!
    var receivedAudio: RecordedAudio!

    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        audioPlayer = AVAudioPlayer(contentsOfURL: receivedAudio.filePathUrl, error: nil)
        audioPlayer.enableRate = true
        
        audioEngine = AVAudioEngine()
        audioFile = AVAudioFile(forReading: receivedAudio.filePathUrl, error: nil)
    }

    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    
    func clearPreviousSetting() {
        audioEngine.stop()
        audioEngine.reset()
        audioPlayer.stop()
    }
    
    func playAudioWithVariableRate(rate: Float) {
        clearPreviousSetting()
        
        audioPlayer.rate = rate
        audioPlayer.currentTime = 0
        audioPlayer.play()
    }

    
    @IBAction func playSlowAudio(sender: UIButton) {
        playAudioWithVariableRate(0.5)
    }
    
    
    @IBAction func playFastAudio(sender: UIButton) {
        playAudioWithVariableRate(1.5)
    }
    
    
    func playAudioWithVariablePitch(pitch: Float) {
        clearPreviousSetting()
        
        var audioPlayerNode = AVAudioPlayerNode()
        audioEngine.attachNode(audioPlayerNode)
        
        var changePitchEffect = AVAudioUnitTimePitch()
        changePitchEffect.pitch = pitch
        audioEngine.attachNode(changePitchEffect)
        
        audioEngine.connect(audioPlayerNode, to: changePitchEffect, format: nil)
        audioEngine.connect(changePitchEffect, to: audioEngine.outputNode, format: nil)
        
        audioPlayerNode.scheduleFile(audioFile, atTime: nil, completionHandler: nil)
        audioEngine.startAndReturnError(nil)
        
        audioPlayerNode.play()
    }
    
    
    @IBAction func playChipmunkAudio(sender: UIButton) {
        playAudioWithVariablePitch(1000)
    }
    
    
    @IBAction func playDarthvaderAudio(sender: UIButton) {
        playAudioWithVariablePitch(-1000)
    }
    
    
    @IBAction func stopPlayAudio(sender: UIButton) {
        clearPreviousSetting()
    }
    
    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepareForSegue(segue: UIStoryboardSegue, sender: AnyObject?) {
        // Get the new view controller using segue.destinationViewController.
        // Pass the selected object to the new view controller.
    }
    */

}
