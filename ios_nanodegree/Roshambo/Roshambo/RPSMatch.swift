//
//  RPSMatch.swift
//  Roshambo
//
//  Created by SiqiWu on 30/06/2015.
//  Copyright (c) 2015 Siqi. All rights reserved.
//

import Foundation
import UIKit

// The RPSMatch struct stores the results of a match.
// Now we're ready to store multiple matches in an array, so users can track their match history.

struct RPSMatch {
    let p1: RPS
    let p2: RPS
    
    init(p1: RPS, p2: RPS) {
        self.p1 = p1
        self.p2 = p2
    }
    
    var winner: RPS {
        get {
            return p1.defeats(p2) ? p1: p2
        }
    }
    
    var loser: RPS {
        get {
            return p1.defeats(p2) ? p2: p1
        }
    }
    
    var resultImageView: UIImage {
        get {
            return imageForMatch(self)
        }
    }
    
    var messageLabel: String {
        get {
            return messageForMatch(self)
        }
    }
    
    var resultTitle: String {
        get {
            if (self.p1 == self.p2) {
                return "It's a tie!"
            }
            return resultString(self)
        }
    }
    
    var resultDescription: String {
        get {
            return self.p1.description + " vs. " + self.p2.description
        }
    }
    
    
    private func messageForMatch(match: RPSMatch) -> String {
        // Handle the tie
        if match.p1 == match.p2 {
            return "It's a tie!"
        }
        
        // Here we build up the results message "RockCrushesScissors. You Win!" etc.
        return match.winner.description + " " + victoryModeString(match.winner) + " " + match.loser.description + ". " + resultString(match)
    }
    
    private func victoryModeString(gesture: RPS) -> String {
        switch(gesture) {
        case .Rock:
            return "crushes"
        case .Scissors:
            return "cuts"
        case .Paper:
            return "covers"
        }
    }
    
    private func resultString(match: RPSMatch) -> String {
        return match.p1.defeats(match.p2) ? "You Win!" : "You Lose!"
    }
    
    private func imageForMatch(match: RPSMatch) -> UIImage {
        var name = ""
        
        switch (match.winner) {
        case .Rock:
            name = "RockCrushesScissors"
        case .Paper:
            name = "PaperCoversRock"
        case .Scissors:
            name = "ScissorsCutPaper"
        }
        
        if match.p1 == match.p2 {
            name = "itsATie"
        }
        return resizeImage(name)
    }
    
    private func resizeImage(imageName: String) -> UIImage {
        let image = UIImage(named: imageName)!
        let newSize = CGSize(width: 240,height: 180)
        UIGraphicsBeginImageContextWithOptions(newSize, false, CGFloat(1.0))
        image.drawInRect(CGRect(origin: CGPointZero, size: newSize))
        let scaledImage = UIGraphicsGetImageFromCurrentImageContext()
        return scaledImage
    }
}