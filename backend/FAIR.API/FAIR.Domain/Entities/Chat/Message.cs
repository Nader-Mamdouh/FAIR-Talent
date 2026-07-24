using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;
using FAIR.Domain.Entities.Identity;

namespace FAIR.Domain.Entities.Chat
{
    public class Message
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();

        [Required]
        public string? Content { get; set; }

        [Required]
        public string? SenderId { get; set; }

        // This will be populated from the Sender's username
        public string? SenderName { get; set; }

        // For private messages
        public string? ReceiverId { get; set; }

        // For group messages
        public string? GroupId { get; set; }

        public DateTime CreateData { get; set; }

        public bool IsRead { get; set; }


        // Navigation properties
        [ForeignKey(nameof(SenderId))]
        public AppUser? Sender { get; set; }

        [ForeignKey(nameof(ReceiverId))]
        public AppUser? Receiver { get; set; }
    }
}
