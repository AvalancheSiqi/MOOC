//
//  RecordSoundsViewController.swift
//  Pitch Perfect
//
//  Created by SiqiWu on 25/06/2015.
//  Copyright (c) 2015 Siqi. All rights reserved.
//

import UIKit
import AVFoundation

class RecordSoundsViewController: UIViewController, AVAudioRecorderDelegate {
    
    @IBOutlet weak var recordBtn: UIButton!
    @IBOutlet weak var stopBtn: UIButton!
    @IBOutlet weak var recordInProgress: UILabel!
    @IBOutlet weak var tapToRecord: UILabel!
    
    var audioRecorder: AVAudioRecorder!
    var recordedAudio: RecordedAudio!

    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }
    
    
    override func viewWillAppear(animated: Bool) {
        // Enable record button and disable stop button
        recordBtn.enabled = true
        stopBtn.hidden = true
    }

    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    
    @IBAction func recordAudio(sender: UIButton) {
        // Change the views
        tapToRecord.hidden = true
        recordInProgress.hidden = false
        recordBtn.enabled = false
        stopBtn.hidden = false
        
        // Record the user's voice
        let dirPath = NSSearchPathForDirectoriesInDomains(.DocumentDirectory, .UserDomainMask, true)[0] as! String
        let recordingName = "my_audio.wav"
        let pathArray = [dirPath, recordingName]
        let filePath = NSURL.fileURLWithPathComponents(pathArray)
        println(filePath)
        
        var session = AVAudioSession.sharedInstance()
        session.setCategory(AVAudioSessionCategoryPlayAndRecord, error: nil)
        
        audioRecorder = AVAudioRecorder(URL: filePath, settings: nil, error: nil)
        audioRecorder.delegate = self
        audioRecorder.meteringEnabled = true
        audioRecorder.prepareToRecord()
        audioRecorder.record()
        
    }
    
    
    func audioRecorderDidFinishRecording(recorder: AVAudioRecorder!, successfully flag: Bool) {
        if (flag) {
            // Store recorded audio to local file
            recordedAudio = RecordedAudio(filePathUrl: recorder.url!, title: recorder.url.lastPathComponent!)
            
            // Move to the next scene aka perform segue
            self.performSegueWithIdentifier("stopRecording", sender: recordedAudio)
        } else {
            println("Recording was not successful")
            recordBtn.enabled = true
            stopBtn.hidden = true
        }
    }
    
    
    override func prepareForSegue(segue: UIStoryboardSegue, sender: AnyObject?) {
        if (segue.identifier == "stopRecording") {
            let playSoundsVC: PlaySoundsViewController = segue.destinationViewController as! PlaySoundsViewController
            let data = sender as! RecordedAudio
            playSoundsVC.receivedAudio = data
        }
    }
    
    
    @IBAction func stopRecording(sender: UIButton) {
        recordInProgress.hidden = true
        tapToRecord.hidden = false
        
        audioRecorder.stop()
        var audioSession = AVAudioSession.sharedInstance()
        audioSession.setActive(false, error: nil)
    }

}

