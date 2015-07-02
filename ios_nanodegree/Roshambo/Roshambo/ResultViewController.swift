//
//  ResultViewController.swift
//  Roshambo
//
//  Created by SiqiWu on 28/06/2015.
//  Copyright (c) 2015 Siqi. All rights reserved.
//

import UIKit

class ResultViewController: UIViewController {
    
    var match: RPSMatch!
    @IBOutlet var resultImageView: UIImageView!
    @IBOutlet weak var messageLabel: UILabel!
    
    
    override func viewWillAppear(animated: Bool) {
        super.viewWillAppear(animated)
        self.messageLabel.text = match.messageLabel
        self.resultImageView.image = match.resultImageView
    }
    
    @IBAction func playAgain() {
        self.dismissViewControllerAnimated(true, completion: nil)
    }

}
