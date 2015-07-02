//
//  RPSViewController.swift
//  Roshambo
//
//  Created by SiqiWu on 28/06/2015.
//  Copyright (c) 2015 Siqi. All rights reserved.
//

import UIKit

class RPSViewController: UIViewController {

    @IBOutlet weak var scissorsBtn: UIButton!
    @IBOutlet weak var rockBtn: UIButton!
    @IBOutlet weak var paperBtn: UIButton!
    
    var match: RPSMatch!
    var history = [RPSMatch]()
    
    
    @IBAction func makeAMove(sender: UIButton) {
        // The RPS enum holds a player's move
        switch(sender) {
        case self.rockBtn: throwDown(RPS.Rock)
        case self.paperBtn: throwDown(RPS.Paper)
        case self.scissorsBtn: throwDown(RPS.Scissors)
        default:
            assert(false, "An unknown button is invoking makeYourMove()")
        }
    }
    
    private func throwDown(playersMove: RPS) {
        let computerMove = RPS()
        self.match = RPSMatch(p1: playersMove, p2: computerMove)
        history.append(match)
        performSegueWithIdentifier("throwDownMove", sender: self)
    }
    
    
    @IBAction func showHistory(sender: UIButton) {
        performSegueWithIdentifier("showHistory", sender: self)
    }
    
    
    // MARK: Segue
    override func prepareForSegue(segue: UIStoryboardSegue, sender: AnyObject?) {
        if segue.identifier == "throwDownMove" {
            let controller = segue.destinationViewController as! ResultViewController
            controller.match = self.match
        }
        if segue.identifier == "showHistory" {
            let controller = segue.destinationViewController as! HistoryViewController
            controller.history = self.history
        }
    }


}

